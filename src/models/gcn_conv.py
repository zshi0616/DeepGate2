import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
import torch_geometric as tg
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_add
from torch_geometric.nn.glob import *
from torch_geometric.nn import MessagePassing


class AggConv(MessagePassing):
    '''
    Modified based on GCNConv implementation in PyG.
    https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
    '''
    def __init__(self, in_channels, ouput_channels=None, wea=False, mlp=None, reverse=False):
        super().__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')  # "Add" aggregation (Step 5).
        if ouput_channels is None:
            ouput_channels = in_channels
        assert (in_channels > 0) and (ouput_channels > 0), 'The dimension for the AggConv should be larger than 0.'

        self.wea = wea

        self.msg = nn.Linear(in_channels, ouput_channels) if mlp is None else mlp

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr=None):
        # TODO: add the normalization part like AggConv
        # x_j has shape [E, dim_emb]
        if self.wea:
            return self.msg(torch.cat((x_j, edge_attr), dim=1))
        else:
            return self.msg(x_j)

    def update(self, aggr_out):
        return aggr_out