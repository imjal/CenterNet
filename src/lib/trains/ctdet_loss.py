from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import time
from models.losses import FocalLoss, RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, CrossEntropy2d
from models.decode import ctdet_decode
from models.utils import _sigmoid
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
      hm_loss += self.crit(output['hm'], batch['hm'], batch['unconf_hm']) / opt.num_stacks
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