from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from utils.dag_utils import subgraph, custom_backward_subgraph
from utils.utils import generate_hs_init

from .mlp import MLP
from .mlp_aggr import MlpAggr

from torch.nn import LSTM, GRU

_update_function_factory = {
    'lstm': LSTM,
    'gru': GRU,
}

class MLPGate_Merge(nn.Module):
    '''
    Recurrent Graph Neural Networks for Circuits.
    '''
    def __init__(self, args):
        super(MLPGate_Merge, self).__init__()
        
        self.args = args

         # configuration
        self.num_rounds = args.num_rounds
        self.device = args.device
        self.predict_diff = args.predict_diff
        self.intermediate_supervision = args.intermediate_supervision
        self.reverse = args.reverse
        self.custom_backward = args.custom_backward
        self.use_edge_attr = args.use_edge_attr
        self.mask = args.mask

        # dimensions
        self.num_aggr = args.num_aggr
        self.dim_node_feature = args.dim_node_feature
        self.dim_hidden = args.dim_hidden
        self.dim_mlp = args.dim_mlp
        self.dim_pred = args.dim_pred
        self.num_fc = args.num_fc
        self.wx_update = args.wx_update
        self.wx_mlp = args.wx_mlp
        self.dim_edge_feature = args.dim_edge_feature

        # Network 
        self.aggr_and = MlpAggr(self.dim_hidden*1, args.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')
        self.aggr_not = MlpAggr(self.dim_hidden*1, args.dim_mlp, self.dim_hidden, num_layer=3, act_layer='relu')
        self.update_and = GRU(self.dim_hidden, self.dim_hidden)
        self.update_not = GRU(self.dim_hidden, self.dim_hidden)

        # Readout 
        self.readout_prob = MLP(self.dim_hidden, args.dim_mlp, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')
        self.readout_rc = MLP(self.dim_hidden * 2, args.dim_mlp, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', sigmoid=True)

        # consider the embedding for the LSTM/GRU model initialized by non-zeros
        self.one = torch.ones(1).to(self.device)
        # self.hs_emd_int = nn.Linear(1, self.dim_hidden)
        self.hf_emd_int = nn.Linear(1, self.dim_hidden)
        self.one.requires_grad = False

    def forward(self, G):
        num_nodes = G.num_nodes
        num_layers_f = max(G.forward_level).item() + 1
        num_layers_b = max(G.backward_level).item() + 1
        
        # initialize the hidden state
        if self.args.disable_encode: 
            h_init = torch.zeros(num_nodes, self.dim_hidden)
            max_sim = 0
            h_init = h_init.to(self.device)
        else:
            h_init = torch.zeros(num_nodes, self.dim_hidden)
            h_init, max_sim = generate_hs_init(G, h_init, self.dim_hidden)
            h_init = h_init.to(self.device)
        
        preds = self._gru_forward(G, h_init, num_layers_f, num_layers_b)
        
        return preds, max_sim
            
    def _gru_forward(self, G, h_init, num_layers_f, num_layers_b, h_true=None, h_false=None):
        G = G.to(self.device)
        x, edge_index = G.x, G.edge_index
        edge_attr = G.edge_attr if self.use_edge_attr else None

        node_state = h_init.to(self.device)
        and_mask = G.gate.squeeze(1) == 1
        not_mask = G.gate.squeeze(1) == 2

        for _ in range(self.num_rounds):
            for level in range(1, num_layers_f):
                # forward layer
                layer_mask = G.forward_level == level

                # AND Gate
                l_and_node = G.forward_index[layer_mask & and_mask]
                if l_and_node.size(0) > 0:
                    and_edge_index, and_edge_attr = subgraph(l_and_node, edge_index, edge_attr, dim=1)
                    # Update structure hidden state
                    msg = self.aggr_and(node_state, and_edge_index, and_edge_attr)
                    and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    hs_and = torch.index_select(node_state, dim=0, index=l_and_node)
                    _, hs_and = self.update_and(and_msg.unsqueeze(0), hs_and.unsqueeze(0))
                    node_state[l_and_node, :] = hs_and.squeeze(0)

                # NOT Gate
                l_not_node = G.forward_index[layer_mask & not_mask]
                if l_not_node.size(0) > 0:
                    not_edge_index, not_edge_attr = subgraph(l_not_node, edge_index, edge_attr, dim=1)
                    # Update structure hidden state
                    msg = self.aggr_not(node_state, not_edge_index, not_edge_attr)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    hs_not = torch.index_select(node_state, dim=0, index=l_not_node)
                    _, hs_not = self.update_not(not_msg.unsqueeze(0), hs_not.unsqueeze(0))
                    node_state[l_not_node, :] = hs_not.squeeze(0)

        node_embedding = node_state.squeeze(0)

        # Readout
        prob = self.readout_prob(node_embedding)
        rc_emb = torch.cat([node_embedding[G.rc_pair_index[0]], node_embedding[G.rc_pair_index[1]]], dim=1)
        is_rc = self.readout_rc(rc_emb)

        return [], node_embedding, prob, is_rc

    
    def imply_mask(self, G, h, h_true, h_false):
        true_mask = (G.mask == 1.0).unsqueeze(0)
        false_mask = (G.mask == 0.0).unsqueeze(0)
        normal_mask = (G.mask == -1.0).unsqueeze(0)
        h_mask = h * normal_mask + h_true * true_mask + h_false * false_mask
        return h_mask



def get_mlp_gate_merged(args):
    return MLPGate_Merge(args)