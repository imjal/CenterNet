from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter
import numpy as np
from models.decode import ctdet_filt_centers
from scipy.spatial import distance
import pdb
from copy import deepcopy



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

  def get_ap(recalls, precisions):
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    # compute the precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap


  def mAP(self, batch, outputs, center_thresh):
    predictions = ctdet_filt_centers(outputs['hm'], outputs['reg']) # is this sorted? 
    gt_inds = np.where(batch['hm'].to("cpu").numpy() == 1.0)
    gt_checked = np.zeros((len(gt_inds)))
    filt_pred = []
    for i in range(len(predictions)):
      if predictions[i][2] >= center_thresh:
        filt_pred += [predictions[i]]

    nd = len(filt_pred)
    tp = np.zeros((nd))
    fp = np.zeros((nd))
    for i, p in enumerate(filt_pred):
      x = p[0]
      y = p[1]
      s = p[2]
      min_dist = -np.inf
      min_arg = -1
      pdb.set_trace()
      if len(gt_inds) > 0:
        # dist from each point
        dist = distance.cdist(gt_inds, np.array([[x, y]]), 'euclidean')
        # take min & assign
        min_dist = np.min(dist, axis = 0)
        min_arg = np.argmin(dist, axis = 0)
        if gt_checked[min_arg] == 0:
          tp[i] = 1.
          gt_checked[jmax, t] = 1
        else:
          fp[i] = 1.
    pdb.set_trace()
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)
    recalls = tp / float(len(gt_inds))
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = get_ap(recalls, precisions)

    return recalls, precisions, ap

  def meanIOU(self, batch, outputs):
    def convert2d(arr): # convert a one-hot into a thresholded array
      max_arr = arr.max(axis=0)
      new_arr = arr.argmax(axis = 0) + 1
      new_arr[max_arr < 0.1] = 0
      return new_arr
    # add to conf matrix for each image
    pred = convert2d(outputs['hm'].detach().cpu().numpy()[0])
    gt = convert2d(batch['hm'].detach().cpu().numpy()[0])
    N = batch['hm'].detach().cpu().numpy()[0].shape[0] + 1
    conf_matrix = np.bincount(N * pred.reshape(-1) + gt.reshape(-1), minlength=N ** 2).reshape(N, N)
    
    acc = np.full(N, np.nan, dtype=np.float)
    iou = np.full(N, np.nan, dtype=np.float)
    tp = conf_matrix.diagonal().astype(np.float)
    pos_gt = np.sum(conf_matrix, axis=0).astype(np.float)
    class_weights = pos_gt / np.sum(pos_gt)
    pos_pred = np.sum(conf_matrix, axis=1).astype(np.float)
    acc_valid = pos_gt > 0
    acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
    iou_valid = (pos_gt + pos_pred) > 0
    union = pos_gt + pos_pred - tp
    iou[acc_valid] = tp[acc_valid] / union[acc_valid]
    macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
    miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
    fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
    pacc = np.sum(tp) / np.sum(pos_gt)

    res = {}
    class_names = [
        "__ignore__",
        "person", 
        "4-wheeler", 
        "2-wheeler"
    ]
    res["mIoU"] = 100 * miou
    res["fwIoU"] = 100 * fiou
    for i, name in enumerate(class_names):
        res["IoU-{}".format(name)] = 100 * iou[i]
    res["mACC"] = 100 * macc
    res["pACC"] = 100 * pacc
    for i, name in enumerate(class_names):
        res["ACC-{}".format(name)] = 100 * acc[i]
    return miou


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

    delta_max = 64
    delta_min = 8
    vid_length = 1212
    delta = delta_min
    umax = 64
    a_thresh = 0.75

    replay_buffer = [None] * opt.replay_samples
    replay_idx = 0


    def run_model(batch):
      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True) # send batch to gpu
      output, loss, loss_stats = model_with_loss(batch) # run model 
      loss = loss.mean() # mean of loss
      return output, loss, loss_stats
      
    def update_model(loss):
      if phase == 'train': # if training, update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    for iter_id, batch in enumerate(data_loader): # go through each item in the batch
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)
      replay_buffer[replay_idx] = deepcopy(batch)
      replay_idx = (replay_idx + 1) % opt.replay_samples
      if iter_id == 0:
        for i in range(1, opt.replay_samples):
          replay_buffer[i] = deepcopy(batch)
          replay_idx = (i + 1) % opt.replay_samples
      
      batch_in = batch
      for i in range(1, opt.replay_samples):
        for k, v in replay_buffer[i].items():
          if k == 'meta':
            for kk, vv in v.items():
              batch_in[k][kk] = torch.cat([batch_in[k][kk], vv])
          else:
            batch_in[k] = torch.cat([batch_in[k], v])
      batch = batch_in

      # JITNet logic
      if iter_id % delta == 0:
        u = 0
        update = True
        while(update):
          output, loss, loss_stats = run_model(batch)
          # save the stuff every iteration
          acc = self.meanIOU(batch, output) # self.mAP(batch, output, self.opt.center_thresh)
          if u < umax and acc < a_thresh:
            print(acc)
            update_model(loss)
          else:
            update = False
          u+=1
        if acc > a_thresh:
          delta = min(delta_max, 2 * delta)
        else:
          delta = max(delta_min, delta / 2)
      else: 
        output, loss, loss_stats = run_model(batch)
      
      batch_time.update(time.time() - end)
      end = time.time()
      if opt.debug > 0:
        self.debug(batch, output, iter_id)
      
      # add a bunch of stuff to the bar to print
      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td) # add to the progress bar
      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['input'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg) # update average loss stats
      if not opt.hide_data_time:
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
      else:
        bar.next()
      
      if opt.test:
        self.save_result(output, batch, results)
      del output, loss, loss_stats
    
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