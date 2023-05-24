from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import math
import copy
import torch
import random
import numpy as np

from .circuit_utils import random_pattern_generator, logic

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

def zero_normalization(x):
    mean_x = torch.mean(x)
    std_x = torch.std(x)
    z_x = (x - mean_x) / std_x
    return z_x

class custom_DataParallel(nn.parallel.DataParallel):
# define a custom DataParallel class to accomodate igraph inputs
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(custom_DataParallel, self).__init__(module, device_ids, output_device, dim)

    def scatter(self, inputs, kwargs, device_ids):
        # to overwride nn.parallel.scatter() to adapt igraph batch inputs
        G = inputs[0]
        scattered_G = []
        n = math.ceil(len(G) / len(device_ids))
        mini_batch = []
        for i, g in enumerate(G):
            mini_batch.append(g)
            if len(mini_batch) == n or i == len(G)-1:
                scattered_G.append((mini_batch, ))
                mini_batch = []
        return tuple(scattered_G), tuple([{}]*len(scattered_G))

def collate_fn(G):
    return [copy.deepcopy(g) for g in G]

def pyg_simulation(g, pattern=[]):
    # PI, Level list
    max_level = 0
    PI_indexes = []
    fanin_list = []
    for idx, ele in enumerate(g.forward_level):
        level = int(ele)
        fanin_list.append([])
        if level > max_level:
            max_level = level
        if level == 0:
            PI_indexes.append(idx)
    level_list = []
    for level in range(max_level + 1):
        level_list.append([])
    for idx, ele in enumerate(g.forward_level):
        level_list[int(ele)].append(idx)
    # Fanin list 
    for k in range(len(g.edge_index[0])):
        src = g.edge_index[0][k]
        dst = g.edge_index[1][k]
        fanin_list[dst].append(src)
    
    ######################
    # Simulation
    ######################
    y = [0] * len(g.x)
    if len(pattern) == 0:
        pattern = random_pattern_generator(len(PI_indexes))
    j = 0
    for i in PI_indexes:
        y[i] = pattern[j]
        j = j + 1
    for level in range(1, len(level_list), 1):
        for node_idx in level_list[level]:
            source_signals = []
            for pre_idx in fanin_list[node_idx]:
                source_signals.append(y[pre_idx])
            if len(source_signals) > 0:
                if int(g.x[node_idx][1]) == 1:
                    gate_type = 1
                elif int(g.x[node_idx][2]) == 1:
                    gate_type = 5
                else:
                    raise("This is PI")
                y[node_idx] = logic(gate_type, source_signals)

    # Output
    if len(level_list[-1]) > 1:
        raise('Too many POs')
    return y[level_list[-1][0]], pattern

def get_function_acc(g, node_emb):
    MIN_GAP = 0.05
    # Sample
    retry = 10000
    tri_sample_idx = 0
    correct = 0
    total = 0
    while tri_sample_idx < 100 and retry > 0:
        retry -= 1
        sample_pair_idx = torch.LongTensor(random.sample(range(len(g.tt_pair_index[0])), 2))
        pair_0 = sample_pair_idx[0]
        pair_1 = sample_pair_idx[1]
        pair_0_gt = g.tt_dis[pair_0]
        pair_1_gt = g.tt_dis[pair_1]
        if pair_0_gt == pair_1_gt:
            continue
        if abs(pair_0_gt - pair_1_gt) < MIN_GAP:
            continue

        total += 1
        tri_sample_idx += 1
        pair_0_sim = torch.cosine_similarity(node_emb[g.tt_pair_index[0][pair_0]].unsqueeze(0), node_emb[g.tt_pair_index[1][pair_0]].unsqueeze(0), eps=1e-8)
        pair_1_sim = torch.cosine_similarity(node_emb[g.tt_pair_index[0][pair_1]].unsqueeze(0), node_emb[g.tt_pair_index[1][pair_1]].unsqueeze(0), eps=1e-8)
        pair_0_predDis = 1 - pair_0_sim
        pair_1_predDis = 1 - pair_1_sim
        succ = False
        if pair_0_gt > pair_1_gt and pair_0_predDis > pair_1_predDis:
            succ = True
        elif pair_0_gt < pair_1_gt and pair_0_predDis < pair_1_predDis:
            succ = True
        if succ:
            correct += 1

    if total > 0:
        acc = correct * 1.0 / total
        return acc
    return -1
            
def generate_orthogonal_vectors(n, dim):
    # Generate an initial random vector
    v0 = np.random.randn(dim)
    v0 /= np.linalg.norm(v0)

    # Generate n-1 additional vectors
    vectors = [v0]
    for i in range(n-1):
        # Generate a random vector
        v = np.random.randn(dim)

        # Project the vector onto the subspace spanned by the previous vectors
        for j in range(i+1):
            v -= np.dot(v, vectors[j]) * vectors[j]

        # Normalize the vector
        v /= np.linalg.norm(v)

        # Append the vector to the list
        vectors.append(v)

    # calculate the max cosine similarity between any two vectors
    max_cos_sim = 0
    for i in range(n):
        for j in range(i+1, n):
            vi = vectors[i]
            vj = vectors[j]
            cos_sim = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
            if cos_sim > max_cos_sim:
                max_cos_sim = cos_sim

    return vectors, max_cos_sim

def generate_hs_init(G, hs, no_dim):
    max_sim = 0
    if G.batch == None:
        batch_size = 1
    else:
        batch_size = G.batch.max().item() + 1
    for batch_idx in range(batch_size):
        if G.batch == None:
            pi_mask = (G.forward_level == 0)
        else:
            pi_mask = (G.batch == batch_idx) & (G.forward_level == 0)
        pi_node = G.forward_index[pi_mask]
        pi_vec, batch_max_sim = generate_orthogonal_vectors(len(pi_node), no_dim)
        if batch_max_sim > max_sim:
            max_sim = batch_max_sim
        hs[pi_node] = torch.tensor(pi_vec, dtype=torch.float)
    
    return hs, max_sim
