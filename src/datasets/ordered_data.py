from torch_geometric.data import Data

class OrderedData(Data):
    def __init__(self, edge_index=None, x=None, y=None, \
                 tt_pair_index=None, tt_dis=None, min_tt_dis=None, \
                 rc_pair_index=None, is_rc=None, \
                 forward_level=None, forward_index=None, backward_level=None, backward_index=None):
        super().__init__()
        self.edge_index = edge_index
        self.tt_pair_index = tt_pair_index
        self.x = x
        self.y = y
        self.tt_dis = tt_dis
        self.min_tt_dis = min_tt_dis
        self.forward_level = forward_level
        self.forward_index = forward_index
        self.backward_level = backward_level
        self.backward_index = backward_index
        self.rc_pair_index = rc_pair_index
        self.is_rc = is_rc
    
    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'forward_index' or key == 'backward_index':
            return 0
        elif key == "edge_index" or key == 'tt_pair_index' or key == 'rc_pair_index':
            return 1
        else:
            return 0
