from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from itertools import chain
import pdb


def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv, opt.track_feature)
  if opt.freeze:
    for param in chain(model.base.parameters(), model.dla_up.parameters()):
      param.requires_grad = False
  if opt.optimizer == 'RMSProp':
    optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), opt.lr, momentum=opt.momentum)
  else:
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1,
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.dataset == 'bddstream':
    train_loader = torch.utils.data.DataLoader(Dataset(opt, 'train')) # , pin_memory=True)
  else:
    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'), 
        shuffle=True,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers, 
        pin_memory=True,
        drop_last=True
    )
  
  if opt.test:
    _, preds = trainer.val(0, val_loader) # val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  best = 1e10
  print('Starting training...')
  log_dict_train, results, save_dict = trainer.train(0, train_loader)
  train_loader.dataset.save_results(results, save_dict, opt.save_dir)
  train_loader.dataset.run_online_eval(save_dict)

  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)