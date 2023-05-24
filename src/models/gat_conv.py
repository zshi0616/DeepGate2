import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax
from torch_geometric.nn.glob import *
from torch_geometric.nn import MessagePassing

from .mlp import MLP



class AGNNConv(MessagePassing):
    '''
    Additive form of GAT from DAGNN paper.

    In order to do the fair comparison with DeepSet. I add a FC-based layer before doing the attention.
    '''
    def __init__(self, in_channels, ouput_channels=None, wea=False, mlp=None, reverse=False):
        super(AGNNConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')
        if ouput_channels is None:
            ouput_channels = in_channels
        assert (in_channels > 0) and (ouput_channels > 0), 'The dimension for the DeepSetConv should be larger than 0.'

        self.wea = wea
        if self.wea:
            # fix the size of edge_attributes now
            self.edge_encoder = nn.Linear(16, ouput_channels)

        # linear transformation
        # self.msg = MLP(in_channels, ouput_channels, ouput_channels, num_layer=3, p_drop=0.2) if mlp is None else mlp
        self.msg = nn.Linear(in_channels, ouput_channels)

        # attention
        attn_dim = ouput_channels
        self.attn_lin = nn.Linear(ouput_channels + attn_dim, 1)


    # h_attn_q is needed; h_attn, edge_attr are optional (we just use kwargs to be able to switch node aggregator above)
    def forward(self, x, edge_index, edge_attr=None, **kwargs):

        # Step 2: Linearly transform node feature matrix.
        # h = self.msg(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        # h_i: query, h_j: key, value
        h_attn_q_i = self.msg(x_i)
        h_attn = self.msg(x_j)
        # see comment in above self attention why this is done here and not in forward
        if self.wea:
            edge_embedding = self.edge_encoder(edge_attr)
            h_attn = h_attn + edge_embedding    
        a_j = self.attn_lin(torch.cat([h_attn_q_i, h_attn], dim=-1))
        a_j = softmax(a_j, index, ptr, size_i)
        t = h_attn * a_j
        return t

    def update(self, aggr_out):
        return aggr_out

# '''
# The attention in dot-product form. Modified version of:
# https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.AGNNConv
# '''
# class AGNNConv(MessagePassing):
#     r"""The graph attentional propagation layer from the
#     `"Attention-based Graph Neural Network for Semi-Supervised Learning"
#     <https://arxiv.org/abs/1803.03735>`_ paper

#     .. math::
#         \mathbf{X}^{\prime} = \mathbf{P} \mathbf{X},

#     where the propagation matrix :math:`\mathbf{P}` is computed as

#     .. math::
#         P_{i,j} = \frac{\exp( \beta \cdot \cos(\mathbf{x}_i, \mathbf{x}_j))}
#         {\sum_{k \in \mathcal{N}(i)\cup \{ i \}} \exp( \beta \cdot
#         \cos(\mathbf{x}_i, \mathbf{x}_k))}

#     with trainable parameter :math:`\beta`.

#     Args:
#         requires_grad (bool, optional): If set to :obj:`False`, :math:`\beta`
#             will not be trainable. (default: :obj:`True`)
#         add_self_loops (bool, optional): If set to :obj:`False`, will not add
#             self-loops to the input graph. (default: :obj:`True`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.
#     """
#     def __init__(self, dim_emb, reverse=False):
#         super(AGNNConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')

#         self.lin = torch.nn.Linear(dim_emb, dim_emb)


#     def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        
#         x_norm = F.normalize(self.lin(x), p=2., dim=-1)

#         # propagate_type: (x: Tensor, x_norm: Tensor)
#         return self.propagate(edge_index, x=x, x_norm=x_norm, size=None)


#     def message(self, x_j: Tensor, x_norm_i: Tensor, x_norm_j: Tensor,
#                 index: Tensor, ptr: OptTensor,
#                 size_i: Optional[int]) -> Tensor:
        
#         alpha = (x_norm_i * x_norm_j).sum(dim=-1)
#         alpha = softmax(alpha, index, ptr, size_i)
#         return x_j * alpha.view(-1, 1)

#     def __repr__(self):
#         return '{}()'.format(self.__class__.__name__)
