'''
The code from https://gist.github.com/denisyarats/2074e6f302dc6998a9f6f9051334e3bd
'''
import torch.nn as nn
import torch.nn.init


class LayerNormGRUCell(nn.GRUCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__(input_size, hidden_size, bias)

        self.gamma_ih = nn.Parameter(torch.ones(3 * self.hidden_size))
        self.gamma_hh = nn.Parameter(torch.ones(3 * self.hidden_size))
        self.eps = 0

    def _layer_norm_x(self, x, g, b):
        mean = x.mean(1).expand_as(x)
        std = x.std(1).expand_as(x)
        return g.expand_as(x) * ((x - mean) / (std + self.eps)) + b.expand_as(x)

    def _layer_norm_h(self, x, g, b):
        mean = x.mean(1).expand_as(x)
        return g.expand_as(x) * (x - mean) + b.expand_as(x)

    def forward(self, x, h):

        ih_rz = self._layer_norm_x(
            torch.mm(x, self.weight_ih.narrow(0, 0, 2 * self.hidden_size).transpose(0, 1)),
            self.gamma_ih.narrow(0, 0, 2 * self.hidden_size),
            self.bias_ih.narrow(0, 0, 2 * self.hidden_size))

        hh_rz = self._layer_norm_h(
            torch.mm(h, self.weight_hh.narrow(0, 0, 2 * self.hidden_size).transpose(0, 1)),
            self.gamma_hh.narrow(0, 0, 2 * self.hidden_size),
            self.bias_hh.narrow(0, 0, 2 * self.hidden_size))

        rz = torch.sigmoid(ih_rz + hh_rz)
        r = rz.narrow(1, 0, self.hidden_size)
        z = rz.narrow(1, self.hidden_size, self.hidden_size)

        ih_n = self._layer_norm_x(
            torch.mm(x, self.weight_ih.narrow(0, 2 * self.hidden_size, self.hidden_size).transpose(0, 1)),
            self.gamma_ih.narrow(0, 2 * self.hidden_size, self.hidden_size),
            self.bias_ih.narrow(0, 2 * self.hidden_size, self.hidden_size))

        hh_n = self._layer_norm_h(
            torch.mm(h, self.weight_hh.narrow(0, 2 * self.hidden_size, self.hidden_size).transpose(0, 1)),
            self.gamma_hh.narrow(0, 2 * self.hidden_size, self.hidden_size),
            self.bias_hh.narrow(0, 2 * self.hidden_size, self.hidden_size))

        n = torch.tanh(ih_n + r * hh_n)
        h = (1 - z) * n + z * h
        return h

class LayerNormGRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRU, self).__init__()
        self.cell = LayerNormGRUCell(input_size, hidden_size, bias)
        self.weight_ih_l0 = self.cell.weight_ih
        self.weight_hh_l0 = self.cell.weight_hh
        self.bias_ih_l0 = self.cell.bias_ih
        self.bias_hh_l0 = self.cell.bias_hh

    def forward(self, xs, h):
        h = h.squeeze(0)
        ys = []
        for i in range(xs.size(0)):
            x = xs.narrow(0, i, 1).squeeze(0)
            h = self.cell(x, h)
            ys.append(h.unsqueeze(0))
        y = torch.cat(ys, 0)
        h = h.unsqueeze(0)
        return y, h