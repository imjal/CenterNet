from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import time
from models.losses import FocalLoss, RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, CrossEntropy2d
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer_iter import BaseTrainerIter
from trains.sort import *
import cv2
from .ctdet_loss import CtdetLoss

class ModelWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModelWithLoss, self).__init__()
    self.model = model
    self.loss = loss
    
  def forward(self, batch, is_update=True):
    outputs = self.model(batch['input'])
    if not is_update:
      outputs[-1]['hm'] = _sigmoid(outputs[-1]['hm'])
      return outputs
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats

class CtdetTrainerIter(BaseTrainerIter):
  def __init__(self, opt, model, optimizer=None):
    super(CtdetTrainerIter, self).__init__(opt, model, optimizer=optimizer)
    self.mot_tracker = Sort()
    self.coloursRGB = np.random.randint(256, size=(32,3), dtype=int) #used only for display
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model = ModelWithLoss(model, self.loss)
    
  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
    loss = CtdetLoss(opt)
    return loss_states, loss

  def train_model(self, batch):
      start_time = time.time()
      self.model.train()
      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=self.opt.device, non_blocking=True) # send batch to gpu
      output, loss, loss_stats = self.model(batch)
      loss = loss.mean() # mean of loss
      current_time = time.time()
      model_time = (current_time - start_time)
      return output, loss, loss_stats, model_time

  def run_model(self, batch):
      start_time = time.time()
      self.model.eval() # difference between train and run
      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=self.opt.device, non_blocking=True) # send batch to gpu
      output = self.model(batch, is_update=False) # run model
      current_time = time.time()
      model_time = (current_time - start_time)
      return output[0], model_time
    
  def update_model(self, batch):
      start_time = time.time()
      self.model.train()
      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=self.opt.device, non_blocking=True) # send batch to gpu
      output, loss, loss_stats = self.model(batch)
      loss = loss.mean() # mean of loss
      current_time = time.time()
      model_time = (current_time - start_time)
      start_time = time.time()
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      current_time = time.time()
      update_time = (current_time - start_time)
      return update_time

  def tracking(self, batch, output, iter_id):
    opt = self.opt
    def display_tracker(frame, frame_num, trackers):
      for d in trackers:
          # print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame_num,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]))
          d = d.astype(np.int32)
          color = self.coloursRGB[d[4]%32,:]
          c = (int(color[0]), int(color[1]), int(color[2]))
          frame = cv2.UMat(frame).get()
          cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), c, 2)
          # cv2.rectangle(frame, (int(d[0]), int(d[1] - cat_size[1] - 2)), (int(d[0] + cat_size[0]), int(d[1] - 2)), c, -1)
          # cv2.putText(frame, txt, (int(bbox[0]), int(bbox[1] - 2)), 
          #             font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
      # cv2.imwrite('cv2img.png', frame)
      return frame
    
    img = batch['input'][0].detach().cpu().numpy().transpose(1, 2, 0)
    frame = np.clip(((img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)

    reg = output['reg'] if opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    filt_dets = dets[dets[:, :, 4] > opt.center_thresh]
    trackers = self.mot_tracker.update(filt_dets)
    viz = display_tracker(frame, iter_id, trackers)
    return trackers, viz

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
      if opt.save_video: # only save the predicted and gt images
        return debugger.imgs['out_pred'], debugger.imgs['out_gt']
      
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')

      if opt.task == 'ctdet_semseg':
        debugger.visualize_masks(seg_gt, img_id='out_mask_gt')
        debugger.visualize_masks(seg_pred, img_id='out_mask_pred')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix=iter_id)
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
    self.accum_coco_det.add_det_to_coco(batch['meta']['img_id'].cpu().numpy()[0], dets_out[0])
    # if self.opt.dataset == 'bdd' or self.opt.dataset == 'bddstream':
    #   if 'map_img_id' not in results:
    #     results['map_img_id'] = {}
    #   results['map_img_id'][batch['meta']['img_id'].cpu().numpy()[0]] = batch['meta']['file_name']