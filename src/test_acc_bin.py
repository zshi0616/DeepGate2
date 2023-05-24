from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
from progress.bar import Bar
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from config import get_parse_args
from utils.logger import Logger
from utils.utils import AverageMeter, pyg_simulation
from utils.random_seed import set_seed
from utils.circuit_utils import check_difference
from utils.sat_utils import solve_sat_iteratively
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
from datasets.mlpgate_dataset import MLPGateDataset

MIN_DIST_PROB = 0.5
MIN_DIST_TT = 0.5
NAME_LIST = []
THREHOLD = 0.91

def test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str

    print(args)
    acc_list = []

    dataset = MLPGateDataset(args.data_dir, args)[:100]
    detector = detector_factory['base'](args)

    num_cir = len(dataset)
    print('EXP ID: ', args.exp_id)
    print('Tot num of circuits: ', num_cir)
    
    for ind, g in enumerate(dataset):
        if g.tt_dis.sum() == 0:
            continue
        if len(NAME_LIST) > 0 and g.name not in NAME_LIST:
            continue
        res = detector.run(g)
        hs, hf, prob, is_rc = res['results']
        node_emb = hf
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        tot = 0
        pd_list = []
        gt_list = []

        for pair_index in range(len(g.tt_pair_index[0])):
            pair_A = g.tt_pair_index[0][pair_index]
            pair_B = g.tt_pair_index[1][pair_index]
            pair_gt = g.tt_dis[pair_index]
            pair_pd_sim = torch.cosine_similarity(node_emb[pair_A].unsqueeze(0), node_emb[pair_B].unsqueeze(0), eps=1e-8)
            # Skip 
            if pair_gt != 0 and pair_gt < MIN_DIST_TT:
                continue
            if pair_gt != 0 and abs(g.prob[pair_A] - g.prob[pair_B]) < MIN_DIST_PROB:
                continue
            
            pd_list.append(pair_pd_sim.item())
            gt_list.append(pair_gt.item() == 0)
            tot += 1
            
            if pair_pd_sim > 0.9 and pair_gt > 0:
                tmp_a = 0

        pd_list = np.array(pd_list)
        gt_list = np.array(gt_list)
        fpr, tpr, thresholds = roc_curve(gt_list, pd_list)
        roc_auc = auc(fpr, tpr)
        opt_thro = thresholds[np.argmax(tpr - fpr)]
        # Threshold
        # pd_list_bin = pd_list > THREHOLD
        pd_list_bin = pd_list > opt_thro

        tp = np.sum(pd_list_bin & gt_list)
        tn = np.sum((~pd_list_bin) & (~gt_list))
        fp = np.sum(pd_list_bin & (~gt_list))
        fn = np.sum((~pd_list_bin) & gt_list)
        
        print('Circuit: {}, Size: {:}'.format(
            g.name, len(g.x)
        ))
        print('Threshold: {:.2f}'.format(opt_thro))
        print('TP: {:.2f}%, TN: {:.2f}%, FP: {:.2f}%, FN: {:.2f}%'.format(
            tp / tot * 100, tn / tot * 100, fp / tot * 100, fn / tot * 100
        ))
        print('Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%'.format(
            (tp + tn) / tot * 100, tp / (tp + fp) * 100, tp / (tp + fn) * 100
        ))
        print('F1 Score: {:.3f}'.format(
            2 * tp / (2 * tp + fp + fn)
        ))
        print('AUC: {:.6f}'.format(
            roc_auc
        ))
        print()
    

if __name__ == '__main__':
    args = get_parse_args()
    set_seed(args)

    test(args)
