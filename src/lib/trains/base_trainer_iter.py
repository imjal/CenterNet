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



class ModelWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModelWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch):
    outputs = self.model(batch['input'])
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

    def run_model(batch):
      start_time = time.time()
      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True) # send batch to gpu
      output, loss, loss_stats = model_with_loss(batch) # run model 
      loss = loss.mean() # mean of loss
      current_time = time.time()
      model_time = (current_time - start_time)
      return output, loss, loss_stats, model_time
    
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
            output, loss, loss_stats, tmp_model_time = run_model(batch)
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
        else: 
          output, loss, loss_stats, model_time = run_model(batch)
      else:
        output, loss, loss_stats, model_time = run_model(batch)

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
      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['input'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg) # update average loss stats
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
      else:
        bar.next()
      time_str = 'total {:.3f}s | load {:.3f}s | model_time {:.3f}s | update_time {:.3f}s | display {:.3f}s'.format(tot_time, load_time, model_time, update_time, display_time)
      print(time_str)
      
      if opt.test:
        self.save_result(output, batch, results)
      del output, loss, loss_stats
      iter_id+=1
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
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