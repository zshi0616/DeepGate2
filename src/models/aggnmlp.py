import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn

from typing import Optional
from torch import Tensor
from torch_geometric.utils import softmax
from torch_geometric.typing import Adj, OptTensor
from .mlp import MLP

class AttnMLP(MessagePassing):
    '''
    The message propagation methods described in NeuroSAT (2 layers without dropout) and CircuitSAT (2 layers, dim = 50, dropout - 20%).
    Cite from NeuroSAT:
    `we sum the outgoing messages of each of a node’s neighbors to form the incoming message.`
    '''
    def __init__(self, in_channels, mlp_channels=512, ouput_channels=64, num_layer=3, p_drop=0.2, act_layer=None, norm_layer=None, reverse=False, mlp_post=None):
        super(AttnMLP, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')
        if ouput_channels is None:
            ouput_channels = in_channels
        assert (in_channels > 0) and (ouput_channels > 0), 'The dimension for the DeepSetConv should be larger than 0.'

        self.msg_pre = MLP(in_channels, mlp_channels, ouput_channels, 
                       num_layer=num_layer, p_drop=p_drop, act_layer=act_layer, norm_layer=norm_layer)
        self.msg = nn.Linear(in_channels, ouput_channels)
        self.msg_post = None if mlp_post is None else mlp_post
        self.attn_lin = nn.Linear(ouput_channels + ouput_channels, 1)


        self.msg_q = MLP(in_channels, mlp_channels, ouput_channels, 
                       num_layer=num_layer, p_drop=p_drop, act_layer=act_layer, norm_layer=norm_layer)
        self.msg_k = MLP(in_channels, mlp_channels, ouput_channels, 
                       num_layer=num_layer, p_drop=p_drop, act_layer=act_layer, norm_layer=norm_layer)
        self.msg_v = MLP(in_channels, mlp_channels, ouput_channels, 
                       num_layer=num_layer, p_drop=p_drop, act_layer=act_layer, norm_layer=norm_layer)


    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        # h_i: query, h_j: key, value
        h_attn_q_i = self.msg_q(x_i)
        h_attn = self.msg_k(x_j)
        # see comment in above self attention why this is done here and not in forward
        a_j = self.attn_lin(torch.cat([h_attn_q_i, h_attn], dim=-1))
        a_j = softmax(a_j, index, ptr, size_i)
        t = self.msg_v(x_j) * a_j
        return t
    
    def update(self, aggr_out):
        if self.msg_post is not None:
            return self.msg_post(aggr_out)
        else:
            return aggr_out


