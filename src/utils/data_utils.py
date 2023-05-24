'''
Utility functions for data handling
'''

import torch
import math
import os

import numpy as np


def read_file(file_name):
    f = open(file_name, "r")
    data = f.readlines()
    return data

def write_file(filename, dir, y):
    path = os.path.join(dir,filename)
    f = open(path, "w")
    for val in y:
        f.write(str(val[0]) + "\n")
    f.close()


def read_npz_file(filename, dir):
    path = os.path.join(dir, filename)
    data = np.load(path, allow_pickle=True)
    return data



def write_subcircuits(filename, dir, x_data, edge_data):
    # format : "Node name:Gate Type:Logic Level:C1-3Circuits:C0:O:Fanout:Reconvergence"
    path = os.path.join(dir,filename)
    f = open(path, "w")

    # x_data = x_data.numpy()

    for node in x_data:
        for n in node:
            f.write(str(n) + ":")
        f.write(";")
    f.write("\n")

    for edge in edge_data:
        f.write("(" + str(edge[0]) + "," + str(edge[1]) + ");")
    f.write("\n")
    f.close()



def update_labels(x, y):
    for idx, val in enumerate(x):
        y[idx] = [y[idx][0] - val[3]]

    return y


def remove(initial_sources):
    final_list = []
    for num in initial_sources:
        if num not in final_list:
            final_list.append(num)
    return final_list


def one_hot(idx, length):
    if type(idx) is int:
        idx = torch.LongTensor([idx]).unsqueeze(0)
    else:
        idx = torch.LongTensor(idx).unsqueeze(0).t()
    x = torch.zeros((len(idx), length)).scatter_(1, idx, 1)
    return x



def construct_node_feature(x, no_node_cop, node_reconv, num_gate_types):
    # the one-hot embedding for the gate types
    gate_list = x[:, 1]
    gate_list = np.float32(gate_list)
    x_torch = one_hot(gate_list, num_gate_types)
    # if node_reconv:
    #     reconv = torch.tensor(x[:, 7], dtype=torch.float).unsqueeze(1)
    #     x_torch = torch.cat([x_torch, reconv], dim=1)
    return x_torch


def add_skip_connection(x, edge_index, edge_attr, ehs):
    for (ind, node) in enumerate(x):
        if node[7] == 1:
            d = ind
            s = node[8]
            new_edge =  torch.tensor([s, d], dtype=torch.long).unsqueeze(0)
            edge_index = torch.cat((edge_index, new_edge), dim=0)
            ll_diff = node[2] - x[int(node[8])][2]
            new_attr = add_edge_attr(1, ehs, ll_diff)
            edge_attr = torch.cat([edge_attr, new_attr], dim=0)
    return edge_index, edge_attr


def add_edge_attr(num_edge, ehs, ll_diff=1):
    positional_embeddings = torch.zeros(num_edge, ehs)
    for position in range(num_edge):
        for i in range(0, ehs, 2):
            positional_embeddings[position, i] = (
                math.sin(ll_diff / (10000 ** ((2 * i) / ehs)))
            )
            positional_embeddings[position, i + 1] = (
                math.cos(ll_diff / (10000 ** ((2 * (i + 1)) / ehs)))
            )

    return positional_embeddings
