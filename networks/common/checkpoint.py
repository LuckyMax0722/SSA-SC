from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch
import os
from glob import glob

from networks.common.io_tools import _remove_recursively, _create_directory


def load(model, optimizer, scheduler, resume, path, logger):
  '''
  Load checkpoint file
  '''

  # If not resume, initialize model and return everything as it is
  if not resume:
    logger.info('=> No checkpoint. Initializing model from scratch')
    model.weights_init()
    epoch = 1
    return model, optimizer, scheduler, epoch

  # If resume, check that path exists and load everything to return
  else:
    file_path = glob(os.path.join(path, '*.pth'))[0]
    assert os.path.isfile(file_path), '=> No checkpoint found at {}'.format(path)
    checkpoint = torch.load(file_path)
    epoch = checkpoint.pop('startEpoch')
    if isinstance(model, (DataParallel, DistributedDataParallel)):
      model.module.load_state_dict(checkpoint.pop('model'))
    else:
      model.load_state_dict(checkpoint.pop('model'))
    optimizer.load_state_dict(checkpoint.pop('optimizer'))
    scheduler.load_state_dict(checkpoint.pop('scheduler'))
    logger.info('=> Continuing training routine. Checkpoint loaded at {}'.format(file_path))
    return model, optimizer, scheduler, epoch


def transform_spconv1_spconv2(pretrained_model):
  model_dict = {
    'SegNet.downCntx.conv1.weight',
    'SegNet.downCntx.conv1_2.weight',
    'SegNet.downCntx.conv2.weight',
    'SegNet.downCntx.conv3.weight',
    'SegNet.resBlock2.conv1.weight',
    'SegNet.resBlock2.conv1_2.weight',
    'SegNet.resBlock2.conv2.weight',
    'SegNet.resBlock2.conv3.weight',
    'SegNet.resBlock2.pool.weight',
    'SegNet.resBlock3.conv1.weight',
    'SegNet.resBlock3.conv1_2.weight',
    'SegNet.resBlock3.conv2.weight',
    'SegNet.resBlock3.conv3.weight',
    'SegNet.resBlock3.pool.weight',
    'SegNet.resBlock4.conv1.weight',
    'SegNet.resBlock4.conv1_2.weight',
    'SegNet.resBlock4.conv2.weight',
    'SegNet.resBlock4.conv3.weight',
    'SegNet.resBlock4.pool.weight',
    'SegNet.resBlock5.conv1.weight',
    'SegNet.resBlock5.conv1_2.weight',
    'SegNet.resBlock5.conv2.weight',
    'SegNet.resBlock5.conv3.weight',
    'SegNet.resBlock5.pool.weight',
    'SegNet.upBlock0.trans_dilao.weight',
    'SegNet.upBlock0.conv1.weight',
    'SegNet.upBlock0.conv2.weight',
    'SegNet.upBlock0.conv3.weight',
    'SegNet.upBlock0.up_subm.weight',
    'SegNet.upBlock1.trans_dilao.weight',
    'SegNet.upBlock1.conv1.weight',
    'SegNet.upBlock1.conv2.weight',
    'SegNet.upBlock1.conv3.weight',
    'SegNet.upBlock1.up_subm.weight',
    'SegNet.upBlock2.trans_dilao.weight',
    'SegNet.upBlock2.conv1.weight',
    'SegNet.upBlock2.conv2.weight',
    'SegNet.upBlock2.conv3.weight',
    'SegNet.upBlock2.up_subm.weight',
    'SegNet.upBlock3.trans_dilao.weight',
    'SegNet.upBlock3.conv1.weight',
    'SegNet.upBlock3.conv2.weight',
    'SegNet.upBlock3.conv3.weight',
    'SegNet.upBlock3.up_subm.weight',
    'SegNet.ReconNet.conv1.weight',
    'SegNet.ReconNet.conv1_2.weight',
    'SegNet.ReconNet.conv1_3.weight',
    'SegNet.logits.weight'
  }

  for key in pretrained_model['model'].keys():
      if key in model_dict:
          pretrained_model['model'][key] = pretrained_model['model'][key].permute([4, 0, 1, 2, 3])
  
  return pretrained_model

def load_model(model, filepath, logger):
  '''
  Load checkpoint file
  '''

  # check that path exists and load everything to return
  assert os.path.isfile(filepath), '=> No file found at {}'
  checkpoint = torch.load(filepath)

  # for spconv1 --> spconv2
  checkpoint = transform_spconv1_spconv2(checkpoint)
  
  if isinstance(model, (DataParallel, DistributedDataParallel)):
    model.module.load_state_dict(checkpoint.pop('model'))
  else:
    model.load_state_dict(checkpoint.pop('model'))
  logger.info('=> Model loaded at {}'.format(filepath))
  return model


def save(path, model, optimizer, scheduler, epoch, config):
  '''
  Save checkpoint file
  '''

  # Remove recursively if epoch_last folder exists and create new one
  # _remove_recursively(path)
  _create_directory(path)

  weights_fpath = os.path.join(path, 'weights_epoch_{}.pth'.format(str(epoch).zfill(3)))

  torch.save({
    'startEpoch': epoch+1,  # To start on next epoch when loading the dict...
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'config_dict': config
  }, weights_fpath)

  return weights_fpath