from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter
import numpy as np
from utils.metrics import get_metric, AccumCOCO
from scipy.spatial import distance
import pdb
import copy
import pickle
import matplotlib.pyplot as plt
import cv2
import os


class BaseTrainerIter(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.accum_coco = AccumCOCO()
  
  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.model = DataParallel(
        self.model, device_ids=gpus, 
        chunk_sizes=chunk_sizes).to(device)
    else:
      self.model = self.model.to(device)
    
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader):
    model_with_loss = self.model
    if phase == 'train':
      model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model.module
      model_with_loss.eval()
      torch.cuda.empty_cache() 
    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    
    if opt.save_video:
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      vid_pth = os.path.join(opt.save_dir, opt.exp_id + '_pred')
      gt_pth = os.path.join(opt.save_dir, opt.exp_id + '_gt')
      out_pred = cv2.VideoWriter('{}.mp4'.format(vid_pth),fourcc, 
        opt.save_framerate, (opt.input_w, opt.input_h))
      out_gt = cv2.VideoWriter('{}.mp4'.format(gt_pth),fourcc, 
        opt.save_framerate, (opt.input_w, opt.input_h))

    delta_max = opt.delta_max
    delta_min = opt.delta_min
    delta = delta_min
    umax = opt.umax
    a_thresh = opt.acc_thresh
    metric = get_metric(opt)
    iter_id = 0 

    data_iter = iter(data_loader)
    update_lst = []
    acc_lst = []
    coco_res_lst = []
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
            output, loss, loss_stats, tmp_model_time = self.train_model(batch)
            total_model_time += tmp_model_time
            # save the stuff every iteration
            acc = metric.get_score(batch, output, u)
            print(acc)
            if u < umax and acc < a_thresh:
              update_time = self.update_model(loss)
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
          self.accum_coco.store_metric_coco(iter_id, batch, output, opt)
        else:
          update_lst += [(iter_id, 0)]
          output, model_time = self.run_model(batch)
          if opt.acc_collect and (iter_id % opt.acc_interval == 0):
            acc = metric.get_score(batch, output, 0)
            print(acc)
            acc_lst+=[(iter_id, acc)]
            self.accum_coco.store_metric_coco(iter_id, batch, output, opt)
      else:
        output, model_time = self.run_model(batch)
        if opt.acc_collect:
          acc = metric.get_score(batch, output, 0, is_baseline=True)
          print(acc)
          acc_lst+=[(iter_id, acc)]
          self.accum_coco.store_metric_coco(iter_id, batch, output, opt, is_baseline=True)

      display_start = time.time()

      if opt.tracking: 
        trackers, viz_pred = self.tracking(batch, output, iter_id) # TODO: factor this into the other class
        out_pred.write(viz_pred)
      elif opt.save_video:
        pred, gt = self.debug(batch, output, iter_id)
        out_pred.write(pred)
        out_gt.write(gt)
      if opt.debug > 1:
        self.debug(batch, output, iter_id)

      display_end = time.time()
      display_time = (display_end - display_start)
      end_time = time.time()
      tot_time = (end_time - start_time)

      # add a bunch of stuff to the bar to print
      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td) # add to the progress bar
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
      else:
        bar.next()
      if opt.display_timing:
        time_str = 'total {:.3f}s| load {:.3f}s | model_time {:.3f}s | update_time {:.3f}s | display {:.3f}s'.format(tot_time, load_time, model_time, update_time, display_time)
        print(time_str)
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
    gt_dict = self.accum_coco.get_gt()
    dt_dict = self.accum_coco.get_dt()
    save_dict['gt_dict'] = gt_dict
    save_dict['dt_dict'] = dt_dict
    return ret, results, save_dict
  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError

  def train_model(self, batch):
      raise NotImplementedError

  def run_model(self, batch):
      raise NotImplementedError
    
  def update_model(self, loss):
        raise NotImplementedError

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)