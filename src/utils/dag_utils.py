import random
import torch
import numpy



# see https://github.com/unbounce/pytorch-tree-lstm/blob/66f29a44e98c7332661b57d22501107bcb193f90/treelstm/util.py#L8
# assume nodes consecutively named starting at 0
#
def top_sort(edge_index, graph_size):

    node_ids = numpy.arange(graph_size, dtype=int)

    node_order = numpy.zeros(graph_size, dtype=int)
    unevaluated_nodes = numpy.ones(graph_size, dtype=bool)

    parent_nodes = edge_index[0]
    child_nodes = edge_index[1]

    n = 0
    while unevaluated_nodes.any():
        # Find which parent nodes have not been evaluated
        unevaluated_mask = unevaluated_nodes[parent_nodes]

        # Find the child nodes of unevaluated parents
        unready_children = child_nodes[unevaluated_mask]

        # Mark nodes that have not yet been evaluated
        # and which are not in the list of children with unevaluated parent nodes
        nodes_to_evaluate = unevaluated_nodes & ~numpy.isin(node_ids, unready_children)

        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        n += 1

    return torch.from_numpy(node_order).long()


# to be able to use pyg's batch split everything into 1-dim tensors
def add_order_info_01(graph):

    l0 = top_sort(graph.edge_index, graph.num_nodes)
    ei2 = torch.LongTensor([list(graph.edge_index[1]), list(graph.edge_index[0])])
    l1 = top_sort(ei2, graph.num_nodes)
    ns = torch.LongTensor([i for i in range(graph.num_nodes)])

    graph.__setattr__("_bi_layer_idx0", l0)
    graph.__setattr__("_bi_layer_index0", ns)
    graph.__setattr__("_bi_layer_idx1", l1)
    graph.__setattr__("_bi_layer_index1", ns)

    assert_order(graph.edge_index, l0, ns)
    assert_order(ei2, l1, ns)


def assert_order(edge_index, o, ns):
    # already processed
    proc = []
    for i in range(max(o)+1):
        # nodes in position i in order
        l = o == i
        l = ns[l].tolist()
        for n in l:
            # predecessors
            ps = edge_index[0][edge_index[1] == n].tolist()
            for p in ps:
                assert p in proc
        proc += l


def add_order_info(graph):
    ns = torch.LongTensor([i for i in range(graph.num_nodes)])
    layers = torch.stack([top_sort(graph.edge_index, graph.num_nodes), ns], dim=0)
    ei2 = torch.LongTensor([list(graph.edge_index[1]), list(graph.edge_index[0])])
    layers2 = torch.stack([top_sort(ei2, graph.num_nodes), ns], dim=0)

    graph.__setattr__("bi_layer_index", torch.stack([layers, layers2], dim=0))

def return_order_info(edge_index, num_nodes):
    ns = torch.LongTensor([i for i in range(num_nodes)])
    forward_level = top_sort(edge_index, num_nodes)
    ei2 = torch.LongTensor([list(edge_index[1]), list(edge_index[0])])
    backward_level = top_sort(ei2, num_nodes)
    forward_index = ns
    backward_index = torch.LongTensor([i for i in range(num_nodes)])
    
    return forward_level, forward_index, backward_level, backward_index


def subgraph(target_idx, edge_index, edge_attr=None, dim=0):
    '''
    function from DAGNN
    '''
    le_idx = []
    for n in target_idx:
        ne_idx = edge_index[dim] == n
        le_idx += [ne_idx.nonzero().squeeze(-1)]
    le_idx = torch.cat(le_idx, dim=-1)
    lp_edge_index = edge_index[:, le_idx]
    if edge_attr is not None:
        lp_edge_attr = edge_attr[le_idx, :]
    else:
        lp_edge_attr = None
    return lp_edge_index, lp_edge_attr

def custom_backward_subgraph(l_node, edge_index, device, dim=0):
    '''
    The custom backward subgraph extraction.
    During backwarding, we consider the side inputs of the target nodes as well.

    This function hasn't been checked yet.
    '''

    # Randomly choose one predecessor in edges
    lp_edge_index = torch.Tensor().to(device=device)
    for n in l_node:
        ne_idx = edge_index[dim] == n

        subset_edges = torch.masked_select(edge_index, ne_idx).reshape(edge_index.shape[0], -1)

        pos_count = torch.count_nonzero(ne_idx)
        random_predecessor = random.randint(0, pos_count - 1)

        indices = torch.tensor([random_predecessor], device=device)
        subset_edges = torch.index_select(subset_edges, 1, indices)

        lp_edge_index = torch.cat((lp_edge_index, subset_edges), dim=1)

    lp_edge_index = lp_edge_index.to(torch.long)

    # collect successors of selected (random) predecessor

    updated_edges = lp_edge_index
    for n in l_node:
        n_vec = torch.tensor([n], device=device)

        ne = lp_edge_index[0] == n
        predecessor = lp_edge_index[1][ne]

        se = edge_index[1] == predecessor
        successors = edge_index[0][se]

        for s in successors:

            if s != n:
                s_vec = torch.tensor([s], device=device)
                new_edge = (torch.stack((n_vec, s_vec), dim=0))
                updated_edges = torch.cat((updated_edges, new_edge), dim=1)

    updated_edges = updated_edges.to(torch.long)
    return updated_edges