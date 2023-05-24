import torch
import torch.nn as nn

_norm_layer_factory = {
    'batchnorm': nn.BatchNorm1d,
}

_act_layer_factory = {
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'sigmoid': nn.Sigmoid,
}

class MLP(nn.Module):
  def __init__(self, dim_in=256, dim_hidden=32, dim_pred=1, num_layer=3, norm_layer=None, act_layer=None, p_drop=0.5, sigmoid=False, tanh=False):
    super(MLP, self).__init__()
    '''
    The basic structure is refered from 
    '''
    assert num_layer >= 2, 'The number of layers shoud be larger or equal to 2.'
    if norm_layer in _norm_layer_factory.keys():
        self.norm_layer = _norm_layer_factory[norm_layer]
    if act_layer in _act_layer_factory.keys():
        self.act_layer = _act_layer_factory[act_layer]
    if p_drop > 0:
        self.dropout = nn.Dropout
    
    fc = []
    # 1st layer
    fc.append(nn.Linear(dim_in, dim_hidden))
    if norm_layer:
        fc.append(self.norm_layer(dim_hidden))
    if act_layer:
        fc.append(self.act_layer(inplace=True))
    if p_drop > 0:
        fc.append(self.dropout(p_drop))
    for _ in range(num_layer - 2):
        fc.append(nn.Linear(dim_hidden, dim_hidden))
        if norm_layer:
            fc.append(self.norm_layer(dim_hidden))
        if act_layer:
            fc.append(self.act_layer(inplace=True))
        if p_drop > 0:
            fc.append(self.dropout(p_drop))
    # last layer
    fc.append(nn.Linear(dim_hidden, dim_pred))
    # sigmoid
    if sigmoid:
        fc.append(nn.Sigmoid())
    if tanh:
        fc.append(nn.Tanh())
    self.fc = nn.Sequential(*fc)

  def forward(self, x):
    out = self.fc(x)
    return out