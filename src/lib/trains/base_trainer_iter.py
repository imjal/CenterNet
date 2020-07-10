from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter
import numpy as np
from utils.metrics import get_metric
from scipy.spatial import distance
import pdb
import copy
import pickle
import matplotlib.pyplot as plt



class ModelWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModelWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch, is_update=True):
    outputs = self.model(batch['input'])
    if not is_update:
      return outputs
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats

class BaseTrainerIter(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModelWithLoss(model, self.loss)

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.model_with_loss = DataParallel(
        self.model_with_loss, device_ids=gpus, 
        chunk_sizes=chunk_sizes).to(device)
    else:
      self.model_with_loss = self.model_with_loss.to(device)
    
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)


  def run_epoch(self, phase, epoch, data_loader):
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()

    delta_max = opt.delta_max
    delta_min = opt.delta_min
    delta = delta_min
    umax = opt.umax
    a_thresh = opt.acc_thresh
    metric = get_metric(opt)
    iter_id = 0 

    def train_model(batch):
      start_time = time.time()
      model_with_loss.train()
      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True) # send batch to gpu
      output, loss, loss_stats = model_with_loss(batch) # run model 
      loss = loss.mean() # mean of loss
      current_time = time.time()
      model_time = (current_time - start_time)
      return output, loss, loss_stats, model_time

    def run_model(batch):
      start_time = time.time()
      model_with_loss.eval()
      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True) # send batch to gpu
      output = model_with_loss(batch, is_update=False) # run model
      current_time = time.time()
      model_time = (current_time - start_time)
      return output[0], model_time
    
    def update_model(loss):
      start_time = time.time()
      if phase == 'train': # if training, update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      current_time = time.time()
      update_time = (current_time - start_time)
      return update_time
    
    data_iter = iter(data_loader)
    update_lst = []
    acc_lst = []
    while True:
      load_time, total_model_time, model_time, update_time, tot_time, display_time = 0, 0, 0, 0, 0, 0
      start_time = time.time()
      # data loading
      try:
        batch = next(data_iter)
      except StopIteration:
        break
      
      loaded_time = time.time()
      load_time += (loaded_time - start_time)

      if opt.adaptive:
        if iter_id % delta == 0:
          u = 0
          update = True
          while(update):
            output, loss, loss_stats, tmp_model_time = train_model(batch)
            total_model_time += tmp_model_time
            # save the stuff every iteration
            acc = metric.get_score(batch, output, u)
            print(acc)
            if u < umax and acc < a_thresh:
              update_time = update_model(loss)
            else:
              update = False
            u+=1
          if acc > a_thresh:
            delta = min(delta_max, 2 * delta)
          else:
            delta = max(delta_min, delta / 2)
          model_time = total_model_time / u
          update_lst += [(iter_id, u)]
          acc_lst += [(iter_id, acc)]
        else:
          update_lst += [(iter_id, 0)]
          output, model_time = run_model(batch)
          if opt.acc_collect:
            acc = metric.get_score(batch, output, 0)
            print(acc)
            acc_lst+=[(iter_id, acc)]
      else:
        output, model_time = run_model(batch)
        if opt.acc_collect:
          acc = metric.get_score(batch, output, 0, is_baseline=True)
          print(acc)
          acc_lst+=[(iter_id, acc)]

      display_start = time.time()
      if opt.debug > 0:
        self.debug(batch, output, iter_id)
      display_end = time.time()
      display_time = (display_end - display_start)
      
      end_time = time.time()

      tot_time = (end_time - start_time)

      # add a bunch of stuff to the bar to print
      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td) # add to the progress bar
      # if opt.adaptive:
      #   for l in avg_loss_stats:
      #     avg_loss_stats[l].update(
      #       loss_stats[l].mean().item(), batch['input'].size(0))
      #     Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg) # update average loss stats
      # del loss, loss_stats
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
      else:
        bar.next()
      if opt.display_timing:
        time_str = 'total {:.3f}s| load {:.3f}s | model_time {:.3f}s | update_time {:.3f}s | display {:.3f}s'.format(tot_time, load_time, model_time, update_time, display_time)
        print(time_str)
      
      if opt.test:
        self.save_result(output, batch, results)
      del output
      iter_id+=1
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    save_dict = {}
    if opt.adaptive:
      plt.scatter(*zip(*update_lst))
      plt.xlabel('iteration')
      plt.ylabel('number of updates')
      plt.savefig(opt.save_dir + '/update_frequency.png')
      save_dict['updates'] = update_lst
      plt.clf()
    if opt.acc_collect:
      plt.scatter(*zip(*acc_lst))
      plt.xlabel('iteration')
      plt.ylabel('mAP')
      plt.savefig(opt.save_dir + '/acc_figure.png')
      save_dict['acc'] = acc_lst
    if opt.adaptive and opt.acc_collect:
      x, y = zip(*filter(lambda x: x[1] > 0, update_lst))
      plt.scatter(x, y, c='r', marker='o')
      plt.xlabel('iteration')
    
    # save dict
    pickle.dump(save_dict, open(opt.save_dir + '/raw_save_dict.pkl', 'wb'))
    return ret, results
  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)