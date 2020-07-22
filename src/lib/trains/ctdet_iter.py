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

class CtdetLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.crit_seg = CrossEntropy2d()
    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, wh_loss, off_loss, sem_seg_loss = 0, 0, 0, 0
    for s in range(opt.num_stacks):
      output = outputs[s]
      if not opt.mse_loss:
        output['hm'] = _sigmoid(output['hm'])

      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']
      if opt.eval_oracle_wh:
        output['wh'] = torch.from_numpy(gen_oracle_map(
          batch['wh'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
      if opt.eval_oracle_offset:
        output['reg'] = torch.from_numpy(gen_oracle_map(
          batch['reg'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)
      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
      if opt.wh_weight > 0:
        if opt.dense_wh:
          mask_weight = batch['dense_wh_mask'].sum() + 1e-4
          wh_loss += (
            self.crit_wh(output['wh'] * batch['dense_wh_mask'],
            batch['dense_wh'] * batch['dense_wh_mask']) / 
            mask_weight) / opt.num_stacks
        elif opt.cat_spec_wh:
          wh_loss += self.crit_wh(
            output['wh'], batch['cat_spec_mask'],
            batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
        else:
          wh_loss += self.crit_reg(
            output['wh'], batch['reg_mask'],
            batch['ind'], batch['wh']) / opt.num_stacks
      if opt.task == 'ctdet_semseg':
        sem_seg_loss = torch.mean(self.crit_seg(output['seg'], batch['seg'][0], batch['weight_seg'][0]))
      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                             batch['ind'], batch['reg']) / opt.num_stacks
        
    loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
           opt.off_weight * off_loss + sem_seg_loss
    loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                  'wh_loss': wh_loss, 'off_loss': off_loss, 'sem_seg_loss': sem_seg_loss}
    return loss, loss_stats

class CtdetTrainerIter(BaseTrainerIter):
  def __init__(self, opt, model, optimizer=None):
    super(CtdetTrainerIter, self).__init__(opt, model, optimizer=optimizer)
    self.mot_tracker = Sort()
    self.coloursRGB = np.random.randint(256, size=(32,3), dtype=int) #used only for display
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
    loss = CtdetLoss(opt)
    return loss_states, loss

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
    if self.opt.dataset == 'bdd' or self.opt.dataset == 'bddstream':
      if 'map_img_id' not in results:
        results['map_img_id'] = {}
      results['map_img_id'][batch['meta']['img_id'].cpu().numpy()[0]] = batch['meta']['file_name']