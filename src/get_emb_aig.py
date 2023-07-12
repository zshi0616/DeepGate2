from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
from progress.bar import Bar
import random
import time
import torch
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt

import utils.circuit_utils as circuit_utils
from config import get_parse_args
from utils.utils import AverageMeter, pyg_simulation, get_function_acc
from utils.random_seed import set_seed
from utils.sat_utils import solve_sat_iteratively
from utils.aiger_utils import aig_to_xdata
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
from datasets.mlpgate_dataset import MLPGateDataset
from datasets.load_data import parse_pyg_mlpgate

AIG_DIR = "../dataset/epfl/"
TMP_DIR = "./tmp"
EMB_DIR = "./emb"
NEW_AIG_DIR = './tmp/aig'
AIG_NAMELIST = []

def save_emb(emb, prob, path):
    f = open(path, 'w')
    f.write('{} {}\n'.format(len(emb), len(emb[0])))
    for i in range(len(emb)):
        for j in range(len(emb[i])):
            f.write('{:.6f} '.format(float(emb[i][j])))
        f.write('\n')
    for i in range(len(prob)):
        f.write('{:.6f}\n'.format(float(prob[i])))
    f.close()

def save_aig(aig_src, aig_dst):
    shutil.copy(aig_src, aig_dst)

def test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str
    detector = detector_factory['base'](args)
    if len(AIG_NAMELIST) == 0:
        for filename in glob.glob(os.path.join(AIG_DIR, '*.aig')):
            aig_name = filename.split('/')[-1].split('.')[0]
            AIG_NAMELIST.append(aig_name)

    for aig_name in AIG_NAMELIST:
        aig_filepath = os.path.join(AIG_DIR, aig_name + '.aiger')
        tmp_aag_filepath = os.path.join(TMP_DIR, aig_name + '.aag')
        x_data, edge_index = aig_to_xdata(aig_filepath, tmp_aag_filepath, args.gate_to_index)
        os.remove(tmp_aag_filepath)
        if len(x_data) == 0:
            continue
        fanin_list, fanout_list = circuit_utils.get_fanin_fanout(x_data, edge_index)
        level_list = circuit_utils.get_level(x_data, fanin_list, fanout_list)
        print('Parse AIG: ', aig_filepath)

        # Generate graph 
        x_data = np.array(x_data)
        edge_index = np.array(edge_index)
        tt_dis = []
        min_tt_dis = []
        tt_pair_index = []
        prob = [0] * len(x_data)
        rc_pair_index = [[0, 1]]
        is_rc = []
        g = parse_pyg_mlpgate(
            x_data, edge_index, tt_dis, min_tt_dis, tt_pair_index, prob, rc_pair_index, is_rc, 
            args.use_edge_attr, args.reconv_skip_connection, args.no_node_cop,
            args.node_reconv, args.un_directed, args.num_gate_types,
            args.dim_edge_feature, args.logic_implication, args.mask
        )
        g.to(args.device)

        # Model 
        start_time = time.time()
        res = detector.run(g)
        end_time = time.time()
        hs, hf, prob, is_rc = res['results']
        print("Circuit: {}, Size: {:}, Time: {:.2f}".format(aig_name, len(x_data), end_time-start_time))
        # acc = get_function_acc(g, hf)
        # print("ACC: {:.2f}%".format(acc/100))
        print()

        # Save emb
        emb_filepath = os.path.join(EMB_DIR, aig_name + '.txt')
        save_emb(hf.detach().cpu().numpy(), prob.detach().cpu().numpy(), emb_filepath)
        newaig_filepath = os.path.join(NEW_AIG_DIR, aig_name + '.aig')
        save_aig(aig_filepath, newaig_filepath)

if __name__ == '__main__':
    args = get_parse_args()
    set_seed(args)
    test(args)
