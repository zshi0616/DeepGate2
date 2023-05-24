from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .circuit_dataset import CircuitDataset

dataset_factory = {
  'benchmarks': CircuitDataset,
  'random': CircuitDataset
}