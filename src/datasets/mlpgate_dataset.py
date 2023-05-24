from typing import Optional, Callable, List
import os.path as osp

import torch
import shutil
import os
import copy
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from utils.data_utils import read_npz_file
from .load_data import parse_pyg_mlpgate


class MLPGateDataset(InMemoryDataset):
    r"""
    A variety of circuit graph datasets, *e.g.*, open-sourced benchmarks,
    random circuits.

    Args:
        root (string): Root directory where the dataset should be saved.
        args (object): The arguments specified by the main program.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(self, root, args, transform=None, pre_transform=None, pre_filter=None):
        self.name = 'MIG'
        self.args = args

        assert (transform == None) and (pre_transform == None) and (pre_filter == None), "Cannot accept the transform, pre_transfrom and pre_filter args now."

        # Reload
        inmemory_dir = os.path.join(args.data_dir, 'inmemory')
        if args.reload_dataset and os.path.exists(inmemory_dir):
            shutil.rmtree(inmemory_dir)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        if self.args.small_train:
            name = 'inmemory_small'
        else:
            name = 'inmemory'
        if self.args.no_rc:
            name += '_norc'
        return osp.join(self.root, name)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.args.circuit_file, self.args.label_file]

    @property
    def processed_file_names(self) -> str:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        tot_pairs = 0
        circuits = read_npz_file(self.args.circuit_file, self.args.data_dir)['circuits'].item()
        labels = read_npz_file(self.args.label_file, self.args.data_dir)['labels'].item()
        
        if self.args.small_train:
            subset = 100

        for cir_idx, cir_name in enumerate(circuits):
            print('Parse circuit: {}, {:} / {:} = {:.2f}%'.format(cir_name, cir_idx, len(circuits), cir_idx / len(circuits) * 100))
            x = circuits[cir_name]["x"]
            edge_index = circuits[cir_name]["edge_index"]

            tt_dis = labels[cir_name]['tt_dis']
            min_tt_dis = labels[cir_name]['min_tt_dis']
            tt_pair_index = labels[cir_name]['tt_pair_index']
            prob = labels[cir_name]['prob']

            if self.args.no_rc:
                rc_pair_index = [[0, 1]]
                is_rc = [0]
            else:
                rc_pair_index = labels[cir_name]['rc_pair_index']
                is_rc = labels[cir_name]['is_rc']

            if len(tt_pair_index) == 0 or len(rc_pair_index) == 0:
                print('No tt or rc pairs: ', cir_name)
                continue

            tot_pairs += len(tt_dis)

            # check the gate types
            # assert (x[:, 1].max() == (len(self.args.gate_to_index)) - 1), 'The gate types are not consistent.'
            graph = parse_pyg_mlpgate(
                x, edge_index, tt_dis, min_tt_dis, tt_pair_index, prob, rc_pair_index, is_rc, 
                self.args.use_edge_attr, self.args.reconv_skip_connection, self.args.no_node_cop,
                self.args.node_reconv, self.args.un_directed, self.args.num_gate_types,
                self.args.dim_edge_feature, self.args.logic_implication, self.args.mask
            )
            graph.name = cir_name
            data_list.append(graph)
            if self.args.small_train and cir_idx > subset:
                break

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
        print('Total Circuits: {:} Total pairs: {:}'.format(len(data_list), tot_pairs))

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'