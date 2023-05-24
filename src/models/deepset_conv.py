import torch
from torch_geometric.nn import MessagePassing

from .mlp import MLP

class DeepSetConv(MessagePassing):
    '''
    The message propagation methods described in NeuroSAT (2 layers without dropout) and CircuitSAT (2 layers, dim = 50, dropout - 20%).
    Cite from NeuroSAT:
    `we sum the outgoing messages of each of a node’s neighbors to form the incoming message.`
    '''
    def __init__(self, in_channels, ouput_channels=None, wea=False, mlp=None, reverse=False, mlp_post=None):
        super(DeepSetConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')
        if ouput_channels is None:
            ouput_channels = in_channels
        assert (in_channels > 0) and (ouput_channels > 0), 'The dimension for the DeepSetConv should be larger than 0.'

        self.wea = wea

        self.msg = MLP(in_channels, ouput_channels, ouput_channels, num_layer=3, p_drop=0.2) if mlp is None else mlp
        self.msg_post = None if mlp_post is None else mlp_post


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
        if self.msg_post is not None:
            return self.msg_post(aggr_out)
        else:
            return aggr_out


