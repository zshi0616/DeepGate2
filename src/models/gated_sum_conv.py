import torch
import torch.nn as nn

from torch_geometric.nn.glob import *
from torch_geometric.nn import MessagePassing


class GatedSumConv(MessagePassing):  # dvae needs outdim parameter
    def __init__(self, in_channels, ouput_channels=None, wea=False, mlp=None, reverse=False, mapper=None, gate=None):
        super(GatedSumConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')
        if ouput_channels is None:
            ouput_channels = in_channels
        assert (in_channels > 0) and (ouput_channels > 0), 'The dimension for the Gated Sum should be larger than 0.'

        self.wea = wea

        self.mapper = nn.Linear(in_channels, ouput_channels) if mapper is None else mapper
        self.gate = nn.Sequential(nn.Linear(in_channels, ouput_channels), nn.Sigmoid()) if gate is None else gate

    def forward(self, x, edge_index, edge_attr=None, **kwargs):

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr=None):
        if self.wea:
            h_j = torch.cat((x_j, edge_attr), dim=1)
        else:
            h_j = x_j
        return self.gate(h_j) * self.mapper(h_j)

    def update(self, aggr_out):
        return aggr_out
