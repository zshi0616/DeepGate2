from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# from .recgnn import RecGNNTrainer
# from .convgnn import ConvGNNTrainer
from .base_trainer import BaseTrainer
from .mlpgnn_trainer import MLPGNNTrainer

train_factory = {
  # 'recgnn': RecGNNTrainer,
  'recgnn': BaseTrainer,
  # 'convgnn': ConvGNNTrainer,
  'convgnn': BaseTrainer,
  'dagconvgnn': BaseTrainer,
  'base': BaseTrainer, 
  'mlpgnn': MLPGNNTrainer, 
  'mlpgnn_merge': MLPGNNTrainer
}