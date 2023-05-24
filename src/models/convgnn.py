from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from torch import nn

from .gat_conv import AGNNConv
from .gcn_conv import AggConv
from .deepset_conv import DeepSetConv
from .gated_sum_conv import GatedSumConv
from .mlp import MLP


_aggr_function_factory = {
    'aggnconv': AGNNConv,
    'deepset': DeepSetConv,
    'gated_sum': GatedSumConv,
    'conv_sum': AggConv,
}


class ConvGNN(nn.Module):
    '''
    Convolutional Graph Neural Networks for Circuits.
    '''

    def __init__(self, args):
        super(ConvGNN, self).__init__()

        self.args = args

        # configuration
        self.device = self.args.device
        self.predict_diff = self.args.predict_diff

        # dimensions
        self.num_aggr = args.num_aggr
        self.dim_node_feature = args.dim_node_feature
        self.dim_hidden = args.dim_hidden
        self.dim_mlp = args.dim_mlp
        self.dim_pred = args.dim_pred
        self.num_fc = args.num_fc
        self.wx_mlp = args.wx_mlp

        # 1. message/aggr-related
        if self.args.aggr_function in _aggr_function_factory.keys():
            # TODO: consider the unconsistent dim of node feature and hidden state
            # corner case: gat_conv
            self.aggr = self.aggr_forward = nn.ModuleList([_aggr_function_factory[self.args.aggr_function](self.dim_node_feature, self.dim_hidden) if l == 0 else _aggr_function_factory[self.args.aggr_function](self.dim_hidden) for l in range(self.num_aggr)])
        else:
            raise KeyError('no support {} aggr function.'.format(self.args.aggr_function))

        # 2. predictor-related
        # TODO: support multiple predictors. Use a nn.ModuleList to handle it.
        self.norm_layer = args.norm_layer
        self.activation_layer = args.activation_layer
        if self.wx_mlp:
            self.predictor = MLP(self.dim_hidden+self.dim_node_feature, self.dim_mlp, self.dim_pred, 
            num_layer=self.num_fc, norm_layer=self.norm_layer, act_layer=self.activation_layer, sigmoid=False, tanh=False)
        else:
            self.predictor = MLP(self.dim_hidden, self.dim_mlp, self.dim_pred, 
            num_layer=self.num_fc, norm_layer=self.norm_layer, act_layer=self.activation_layer, sigmoid=False, tanh=False)

    def forward(self, G):
        x, edge_index = G.x, G.edge_index

        preds = []
        for i in range(self.num_aggr):
            x = self.aggr[i](x, edge_index)

        pred = self.predictor(x)
        preds.append(pred)

        return preds


def get_conv_gnn(args):
    return ConvGNN(args)
