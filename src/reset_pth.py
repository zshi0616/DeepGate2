from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from torch_geometric import data

from config import get_parse_args
from models.model import create_model, load_model, save_model
from utils.logger import Logger
from utils.random_seed import set_seed
from utils.circuit_utils import check_difference
from trains.train_factory import train_factory

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

src_pth_filepath = 'exp/prob/aggr_exp_deepset/model_stage1.pth'
dst_pth_filepath = 'exp/prob/aggr_exp_deepset/model_last.pth'

def main(args):
    #################
    # Device 
    #################
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str
    args.device = torch.device('cpu')
    args.world_size = 1
    args.rank = 0  # global rank

    #################
    # Model
    #################
    model = create_model(args)
    if args.local_rank == 0:
        print('==> Creating model...')
        print(model)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    model, optimizer, start_epoch = load_model(
        model, src_pth_filepath, optimizer, args.resume, args.lr, args.lr_step, args.local_rank, args.device)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-4
    
    
    save_model(dst_pth_filepath, 0, model, optimizer)
    print('Load: ', src_pth_filepath)
    print('Save: ', dst_pth_filepath)

if __name__ == '__main__':
    args = get_parse_args()

    main(args)
