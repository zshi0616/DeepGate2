from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from .recgnn import get_recurrent_gnn
from .convgnn import get_conv_gnn
from .dag_convgnn import get_dag_recurrent_gnn
from .mlpgate import get_mlp_gate
from .mlpgate_merge import get_mlp_gate_merged

_model_factory = {
    'recgnn': get_recurrent_gnn,
    'convgnn': get_conv_gnn,
    'dagconvgnn': get_dag_recurrent_gnn,
    'mlpgnn': get_mlp_gate, 
    'mlpgnn_merge': get_mlp_gate_merged, 

}


def create_model(args):
    get_model = _model_factory[args.arch]
    model = get_model(args)
    return model

def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None, local_rank = 0, device='cuda'):
    start_epoch = 0
    # checkpoint = torch.load(
    #   model_path, map_location=lambda storage, loc: storage)

    if device == 'cuda':
        map  = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(
            model_path, map_location=map)
    else:
        checkpoint = torch.load(
        model_path, map_location=lambda storage, loc: storage)
    
    if local_rank == 0:
        print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                if local_rank == 0:
                 print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}. {}'.format(
                          k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            if local_rank == 0:
                print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            if local_rank == 0:
                print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            if local_rank == 0:
                print('Resumed optimizer with start lr', start_lr)
        else:
            if local_rank == 0:
                print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
