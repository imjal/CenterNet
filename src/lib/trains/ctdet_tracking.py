from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
np.set_printoptions(suppress=True)
import time
from models.losses import FocalLoss, RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, CrossEntropy2d
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer_iter import BaseTrainerIter
from .ctdet_loss import CtdetLoss
from trains.sort import *
import cv2
from models.decode import ctdet_decode
from utils.coco import COCO
from utils.image import gaussian_radius
from models.decode import ctdet_top_centers_wh
from .ctdet_loss import CtdetLoss
import pdb
import math
import matplotlib.pyplot as plt
import queue


class Track:
  def __init__(self, track_id, new_feat, pred):
    self.length = 1
    self.track_id = track_id
    self.mean_feat = new_feat
    self.new_feat = new_feat
    self.agg_feat = new_feat
    self.pred = pred
    self.lost = 0
    self.cls = pred[-1]
    # self.feat_queue = torch.zeros((30, 64))
    # self.feat_queue[self.length-1] = new_feat
  
  def update(self, new_feat, pred):
    self.mean_feat = (self.mean_feat * 0.1) + (new_feat * 0.9)
    # self.mean_feat = (self.agg_feat * 0.1) + (new_feat * 0.9)
    # self.agg_feat = (self.agg_feat * self.length + new_feat)/ (self.length + 1)
    # self.new_feat = new_feat
    # TODO: weight the average for newer
    # keep all features around (some large batch 40/50), compute maximum score to match
    # keep the difference of the mean class feature and match them
    # self.feat_queue[self.length-1] = new_feat
    self.length +=1
    self.pred = pred
    self.lost = 0

  def add_lost(self):
    self.lost +=1


class Tracker:
  def __init__(self, opt, t_lost):
    self.opt = opt
    self.t_lost = t_lost
    self.trackers = []
    self.track_idx = 0
    self.cos = nn.CosineSimilarity()

  def extract_features(self, feature_map, filt_pred):
    mean_feat = torch.zeros((len(filt_pred), 64))
    for i in range(len(filt_pred)):
      pred = filt_pred[i]
      x, y = min(pred[0], feature_map.shape[1]-3), min(pred[1], feature_map.shape[2]-3)
      width, height = pred[2], pred[3]
      radius = gaussian_radius((math.ceil(height), math.ceil(width)))
      adj_rad = max(radius/2, 2)
      max_x = feature_map.shape[1]
      max_y = feature_map.shape[2]
      pred_featmap = feature_map[:, max(0, int(x - adj_rad)): min(int(x + adj_rad), max_x), max(0, int(y - adj_rad)): min(max_y, int(y + adj_rad))] # box of size radius, could also do a downsampled version of the box
      mean_feat[i] = pred_featmap.reshape(64, -1).mean(axis = 1)
    return mean_feat

  def display_map(self, batch, tracks, idx):
    opt = self.opt
    for i in range(1):
      debugger = Debugger(opt, dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
     
      debugger.add_img(img, img_id='track')
      for i in range(len(tracks)):
        dets = tracks[i].pred
        bbox = dets[:4] * self.opt.down_ratio
        w, h = bbox[2], bbox[3]
        bbox = np.array([bbox[0] - w/2, bbox[1] - h/2 , bbox[0] + w/2, bbox[1] + h/2])
        debugger.add_coco_bbox(bbox, int(dets[-1]), tracks[i].track_id, img_id='track', tracking=True)

      debugger.save_all_imgs(opt.debug_dir, prefix=f'{idx}')


  def display_sim_map(self, batch, dets, cmpr):
    opt = self.opt
    for i in range(1):
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    implot = plt.imshow(img)
    dets *= self.opt.down_ratio
    plt.scatter(dets[:, 0], dets[:, 1], c=cmpr, cmap ='Reds', s = 50)
    plt.savefig(self.opt.debug_dir + '/{:08d}_track_sim.png'.format(self.idx), dpi=300)
    plt.clf()

  def compare_features(self, new_mean, existing_means):
    '''
    Return the scores where the min score is most similar
    '''
    # process comparisons
    if self.opt.similarity == 'cos':
      results_cmpr = self.cos(new_mean.unsqueeze(0), existing_means)
      return 1 - results_cmpr
    # elif self.opt.similarity == 'maxall':
    #   for i in len(self.trackers):
    #     feature = self.trackers[i].feat_queue[: max(self.trackers[i].length, self.trackers[i].feat_queue.shape[0])]
    else:
      results_cmpr = torch.cdist(new_means, existing_means)
      return results_cmpr[0]

  def get_features(self):
    agg_feat = torch.zeros([len(self.trackers), 64], dtype=torch.float)
    for i in range(len(self.trackers)):
      agg_feat[i] = self.trackers[i].mean_feat
    return agg_feat

  def add_detections(self, dets, features):
    # filter detections
    pred = dets[dets[:, 4] > self.opt.center_thresh]
    features = self.extract_features(features, pred)
    
    # check if there are no tracklets
    if len(self.trackers) == 0:
      self.trackers = [ Track(i, features[i], pred[i]) for i in range(len(pred))]
      self.track_idx = len(pred)
      return self.trackers
    
    # TODO: can do comparisons by class predictions as well, can try that later
    # cmpr = self.compare_features(extracted_means[filt_pred[:, 5] == i], self.mean_feat[self.prev_pred[:, 5] == i])
    unmatched_idx = []
    agg_features = self.get_features()
    visited = [0] * len(self.trackers)
    for i in range(len(pred)):
      cmpr = self.compare_features(features[i], agg_features)
      match_idx = cmpr.argmin().item()
      print(self.trackers[match_idx].track_id, cmpr[match_idx].item())
      if visited[match_idx] or self.trackers[match_idx].cls != pred[i][-1]:
        self.trackers += [Track(self.track_idx, features[i], pred[i])]
        self.track_idx += 1
        continue
      # TODO: check if the matched detection is greater than some threshold of similarity
      visited[match_idx] = 1
      self.trackers[match_idx].update(features[i], pred[i])

    # figure out the unvisited previous detections
    not_visited_indices = list(filter(lambda x: visited[x] == 0, range(len(visited))))
    rem_indices = []
    for x in not_visited_indices:
      self.trackers[x].add_lost()
      if self.trackers[x].lost > self.t_lost:
        rem_indices += [x]
    
    for x in reversed(rem_indices):
      del self.trackers[x]

    return self.trackers


class ModelNoSGD(torch.nn.Module):
  def __init__(self, model, loss, opt):
    super(ModelNoSGD, self).__init__()
    self.model = model
    self.opt = opt
    self.idx = 0
    self.loss = loss
    self.tracker = Tracker(opt, opt.t_lost)
  
  def update(self, is_update): 
    self.is_update = is_update

  def forward(self, batch):
    output = self.model(batch['input'])
    if self.is_update:
      loss, loss_stats = self.loss(output, batch)
      return output[0], loss, loss_stats

    # detach both so pytorch graph doesn't explode
    output[0]['hm'] = _sigmoid(output[0]['hm'])
    features = output[1].detach()
    dets = ctdet_top_centers_wh( output[0]['hm'], output[0]['wh'], reg=output[0]['reg'], cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    predictions = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

    tracks = self.tracker.add_detections(predictions[0], features[0])
    self.tracker.display_map(batch, tracks, self.idx)
    self.idx += 1

    return output[0]

    
class CtdetTracking(BaseTrainerIter):
  def __init__(self, opt, model, optimizer=None):
    super(CtdetTracking, self).__init__(opt, model, optimizer=optimizer)
    self.coloursRGB = np.random.randint(256, size=(32,3), dtype=int) #used only for display
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model = ModelNoSGD(model, self.loss, opt) # move model to upper abstraction since the changes are larger
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
    loss = CtdetLoss(opt)
    return loss_states, loss

  def train_model(self, batch):
      start_time = time.time()
      self.model.model.train()
      self.model.update(True)
      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=self.opt.device, non_blocking=True) # send batch to gpu
      output, loss, loss_stats = self.model(batch)
      loss = loss.mean()
      current_time = time.time()
      model_time = (current_time - start_time)
      return output, loss, None, model_time

  def run_model(self, batch):
      start_time = time.time()
      self.model.model.eval() # difference between train and run
      self.model.update(False)
      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=self.opt.device, non_blocking=True) # send batch to gpu
      output = self.model(batch) # run model
      current_time = time.time()
      model_time = (current_time - start_time)
      return output, model_time
    
  def update_model(self, loss):
      start_time = time.time()
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      current_time = time.time()
      update_time = (current_time - start_time)
      return update_time
      return 0

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.down_ratio
    if opt.task == 'ctdet_semseg':
      seg_gt = batch['seg'][0][0].cpu().numpy()
      seg_pred = output['seg'].max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

    for i in range(1):
      debugger = Debugger(opt, dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      
      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.vis_thresh:
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                 dets[i, k, 4], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.vis_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')

      if opt.save_video and opt.debug <= 1: # only save the predicted and gt images
        return debugger.imgs['out_pred'], debugger.imgs['out_gt'] # , debugger.imgs['pred_hm'], debugger.imgs['gt_hm']
      
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')

      
      if opt.task == 'ctdet_semseg':
        debugger.visualize_masks(seg_gt, img_id='out_mask_gt')
        debugger.visualize_masks(seg_pred, img_id='out_mask_pred')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix=iter_id)
        import pdb; pdb.set_trace()
      
      if opt.save_video:
        return debugger.imgs['out_pred'], debugger.imgs['out_gt']
      # else:
      #   debugger.show_all_imgs(pause=True)
      

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdet_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
    if self.opt.dataset == 'bdd' or self.opt.dataset == 'bddstream':
      if 'map_img_id' not in results:
        results['map_img_id'] = {}
      results['map_img_id'][batch['meta']['img_id'].cpu().numpy()[0]] = batch['meta']['file_name']