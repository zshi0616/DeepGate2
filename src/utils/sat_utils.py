import numpy as np
import random

import torch
from .utils import pyg_simulation


def generate_k_iclause(n, k):
    vs = np.random.choice(n, size=min(n, k), replace=False)
    return [v + 1 if random.random() < 0.5 else -(v + 1) for v in vs]



# the utility function for Circuit-SAT
def get_sub_cnf(cnf, var, is_inv):
    res_cnf = []
    if not is_inv:
        for clause in cnf:
            if not var in clause:
                tmp_clause = clause.copy()
                for idx, ele in enumerate(tmp_clause):
                    if ele == -var:
                        del tmp_clause[idx]
                res_cnf.append(tmp_clause)
    else:
        for clause in cnf:
            if not -var in clause:
                tmp_clause = clause.copy()
                for idx, ele in enumerate(tmp_clause):
                    if ele == var:
                        del tmp_clause[idx]
                res_cnf.append(tmp_clause)
    return res_cnf

def two_fanin_gate(po_idx, fan_in_list, x, edge_index, gate_type):
    gate_list = fan_in_list.copy()
    new_gate_list = []

    while True:
        if len(gate_list) + len(new_gate_list) == 2:
            for gate_idx in gate_list:
                edge_index.append([gate_idx, po_idx])
            for gate_idx in new_gate_list:
                edge_index.append([gate_idx, po_idx])
            break
        if len(gate_list) == 0:
            gate_list = new_gate_list.copy()
            new_gate_list.clear()
        elif len(gate_list) == 1:
            new_gate_list.append(gate_list[0])
            gate_list = new_gate_list.copy()
            new_gate_list.clear()
        else:
            new_gate_idx = len(x)
            x.append(gate_type)
            edge_index.append([gate_list[0], new_gate_idx])
            edge_index.append([gate_list[1], new_gate_idx])
            gate_list = gate_list[2:]
            new_gate_list.append(new_gate_idx)


def save_cnf(cnf, cnf_idx, x, edge_index, inv2idx):
    cnf_fan_in_list = []
    for clause in cnf:
        if len(clause) == 0:
            continue
        elif len(clause) == 1:
            if clause[0] < 0:
                cnf_fan_in_list.append(inv2idx[abs(clause[0])])
            else:
                cnf_fan_in_list.append(clause[0])
        else:
            clause_idx = len(x)
            x.append(one_hot_gate_type('OR'))
            cnf_fan_in_list.append(clause_idx)
            clause_fan_in_list = []
            for ele in clause:
                if ele < 0:
                    clause_fan_in_list.append(inv2idx[abs(ele)])
                else:
                    clause_fan_in_list.append(ele)
            two_fanin_gate(clause_idx, clause_fan_in_list, x, edge_index, x[clause_idx])

    x[cnf_idx] = one_hot_gate_type('AND')
    two_fanin_gate(cnf_idx, cnf_fan_in_list, x, edge_index, x[cnf_idx])

def merge_cnf(cnf):
    res = []
    clause2bool = {}
    for clause in cnf:
        tmp_clause = tuple(clause)
        if not tmp_clause in clause2bool:
            clause2bool[tmp_clause] = True
            res.append(clause)
    return res

def recursion_generation(cnf, cnf_idx, current_depth, max_depth, n_vars, x, edge_index, inv2idx):
    '''
    Expand the CNF as binary tree
    The expanded CNF can be writen as:
        CNF = OR(B_T, B_F)
        B_T = AND(exp_T_CNF, var)
        B_F = AND(exp_F_CNF, var_inv)
        # exp_T_CNF, exp_F_CNF are new CNFs
    Input:
        cnf: iclauses
        cnf_idx: the cnf PO index in x
        current_depth: current expand time
        max_depth: maximum expand time
        n_vars: number of variables
        x: nodes
        edge_index: edge
        inv2idx: PI_inv index
    '''

    ####################
    # Store as CNF
    ####################
    if current_depth == max_depth:
        save_cnf(cnf, cnf_idx, x, edge_index, inv2idx)
        return

    ####################
    # Sort
    ####################
    var_times = [0] * (n_vars + 1)
    for idx in range(1, n_vars + 1, 1):
        for clause in cnf:
            if idx in clause:
                var_times[abs(idx)] += 1

    var_sort = np.argsort(var_times)
    most_var = var_sort[-1]
    if var_times[most_var] == 0:
        save_cnf(cnf, cnf_idx, x, edge_index, inv2idx)
        return


    ####################
    # Expansion
    ####################
    for most_var in var_sort[::-1]:
        var_idx = most_var
        next_var = False
        # Get sub-CNFs
        exp_T_cnf = get_sub_cnf(cnf, most_var, 0)
        exp_F_cnf = get_sub_cnf(cnf, most_var, 1)

        for clause in exp_T_cnf:
            if len(clause) == 0:
                next_var = True
                break
        for clause in exp_F_cnf:
            if len(clause) == 0:
                next_var = True
                break
        if not next_var:
            break
    if most_var == 0:
        save_cnf(cnf, cnf_idx, x, edge_index, inv2idx)
        return

    if not most_var in inv2idx:
        inv2idx[most_var] = len(x)
        x.append(one_hot_gate_type('NOT'))
        edge_index.append([most_var, inv2idx[most_var]])
    var_inv_idx = inv2idx[most_var]

    exp_T_cnf = merge_cnf(exp_T_cnf)
    exp_F_cnf = merge_cnf(exp_F_cnf)

    # ------------------------------------------
    # Construct (exp_T_CNF) and (B_T)
    if len(exp_T_cnf) == 0:
        edge_index.append([var_idx, cnf_idx])
    elif len(exp_T_cnf) == 1:
        # Construct (B_T): B_T = AND(var_idx, exp_T)
        B_T_idx = len(x)
        x.append(one_hot_gate_type('AND'))
        exp_T_cnf = exp_T_cnf[0]
        if len(exp_T_cnf) == 1:  # The clause only have one var
            exp_T_idx = exp_T_cnf[0]
            if exp_T_idx < 0:
                exp_T_idx = inv2idx[abs(exp_T_idx)]
        else:  # The clause have many vars
            exp_T_idx = len(x)
            x.append(one_hot_gate_type('OR'))
            for ele in exp_T_cnf:
                if ele < 0:
                    ele_idx = inv2idx[abs(ele)]
                else:
                    ele_idx = ele
                edge_index.append([ele_idx, exp_T_idx])
        edge_index.append([exp_T_idx, B_T_idx])
        edge_index.append([var_idx, B_T_idx])
        edge_index.append([B_T_idx, cnf_idx])
    else:
        # Construct(exp_T_CNF)
        exp_T_cnf_idx = len(x)
        x.append(one_hot_gate_type('OR'))
        recursion_generation(exp_T_cnf, exp_T_cnf_idx, current_depth + 1, max_depth,
                             n_vars, x, edge_index, inv2idx)
        # Construct (B_T)
        B_T_idx = len(x)
        x.append(one_hot_gate_type('AND'))
        edge_index.append([exp_T_cnf_idx, B_T_idx])
        edge_index.append([var_idx, B_T_idx])
        edge_index.append([B_T_idx, cnf_idx])

    # ------------------------------------------
    # Construct (exp_F_CNF) and (B_F)
    if len(exp_F_cnf) == 0:
        edge_index.append([var_inv_idx, cnf_idx])
    elif len(exp_F_cnf) == 1:
        # Construct (B_F): B_F = AND(var_idx, exp_F)
        B_F_idx = len(x)
        x.append(one_hot_gate_type('AND'))
        exp_F_cnf = exp_F_cnf[0]
        if len(exp_F_cnf) == 1:  # The clause only have one var
            exp_F_idx = exp_F_cnf[0]
            if exp_F_idx < 0:
                exp_F_idx = inv2idx[abs(exp_F_idx)]
        else:  # The clause have many vars
            exp_F_idx = len(x)
            x.append(one_hot_gate_type('OR'))
            for ele in exp_F_cnf:
                if ele < 0:
                    ele_idx = inv2idx[abs(ele)]
                else:
                    ele_idx = ele
                edge_index.append([ele_idx, exp_F_idx])
        edge_index.append([exp_F_idx, B_F_idx])
        edge_index.append([var_inv_idx, B_F_idx])
        edge_index.append([B_F_idx, cnf_idx])
    else:
        # Construct(exp_F_CNF)
        exp_F_cnf_idx = len(x)
        x.append(one_hot_gate_type('OR'))
        recursion_generation(exp_F_cnf, exp_F_cnf_idx, current_depth + 1, max_depth,
                             n_vars, x, edge_index, inv2idx)
        # Construct (B_F)
        B_F_idx = len(x)
        x.append(one_hot_gate_type('AND'))
        edge_index.append([exp_F_cnf_idx, B_F_idx])
        edge_index.append([var_inv_idx, B_F_idx])
        edge_index.append([B_F_idx, cnf_idx])


def one_hot_gate_type(gate_type):
    res = []
    if gate_type == 'PI':
        res = [1, 0, 0, 0]
    elif gate_type == 'AND':        res = [0, 1, 0, 0]
    elif gate_type == 'OR':
        res = [0, 0, 1, 0]
    elif gate_type == 'NOT':
        res = [0, 0, 0, 1]
    else:
        print('[ERROR] Unknown gate type')
    return res



def write_dimacs_to(n_vars, iclauses, out_filename):
    with open(out_filename, 'w') as f:
        f.write("p cnf %d %d\n" % (n_vars, len(iclauses)))
        for c in iclauses:
            for x in c:
                f.write("%d " % x)
            f.write("0\n")


def solve_sat_iteratively(g, detector):
    # here we consider the PO has already been masked 
    print('Name of circuit: ', g.name)
    # print(g)

    # set PO as 1.
    layer_mask = g.backward_level == 0
    l_node = g.backward_index[layer_mask]
    g.mask[l_node] = torch.tensor(1.0)

    # check # PIs
    # literal index
    layer_mask = g.forward_level == 0
    l_node = g.forward_index[layer_mask]
    print('# PIs: ', len(l_node))

    # random solution generation.
    # sol = (torch.rand(len(l_node)) > 0.5).int().unsqueeze(1)
    # sat = pyg_simulation(g, sol)[0]
    # return sol, sat
    # only one forward
    # ret = detector.run(g)
    # output = ret['results'].cpu()
    # sol = (output[l_node] > 0.5).int()
    # sat = pyg_simulation(g, sol)[0]
    # return sol, sat

    # for backtracking
    ORDER = []
    change_ind = -1
    mask_backup = g.mask.clone().detach()


    for i in range(len(l_node)):
        print('==> # ', i+1, 'solving..')
        ret = detector.run(g)
        output = ret['results'].cpu()

        # mask
        one_mask = torch.zeros(g.y.size(0))
        one_mask = one_mask.scatter(dim=0, index=l_node, src=torch.ones(len(l_node))).unsqueeze(1)
        
        max_val, max_ind = torch.max(output * one_mask, dim=0)
        min_val, min_ind = torch.min(output + (1 - one_mask), dim=0)

        ext_val, ext_ind = (max_val, max_ind) if (max_val > (1 - min_val)) else (min_val, min_ind)
        print('Assign No. ', ext_ind.item(), 'with prob: ', ext_val.item(), 'as value: ', 1.0 if ext_val > 0.5 else 0.0)
        # g.mask[ext_ind] = torch.tensor(np.random.binomial(n=1, p=ext_val), dtype=torch.float)
        g.mask[ext_ind] = torch.tensor(1.0) if ext_val > 0.5 else torch.tensor(0.0)
        # push the current index to Q
        ORDER.append(ext_ind)
        
        l_node_new = []
        for i in l_node:
            if i != ext_ind:
                l_node_new.append(i)
        l_node = torch.tensor(l_node_new)
    


    # literal index
    layer_mask = g.forward_level == 0
    l_node = g.forward_index[layer_mask]
    print('Prob: ', output[l_node])
    
    sol = g.mask[l_node]
    print('Solution: ', sol)
    # check the satifiability
    sat = pyg_simulation(g, sol)[0]
    if sat:
        return sol, sat
    
    print('=====> Step into the backtracking...')
    # do the backtracking
    while ORDER:
        # renew the mask
        g.mask = mask_backup.clone().detach()
        change_ind = ORDER.pop()
        print('Change the values when solving No. ', change_ind.item(), 'PIs')
        # literal index
        layer_mask = g.forward_level == 0
        l_node = g.forward_index[layer_mask]

        for i in range(len(l_node)):
            # print('==> # ', i+1, 'solving..')
            ret = detector.run(g)
            output = ret['results'].cpu()

            # mask
            one_mask = torch.zeros(g.y.size(0))
            one_mask = one_mask.scatter(dim=0, index=l_node, src=torch.ones(len(l_node))).unsqueeze(1)
            
            max_val, max_ind = torch.max(output * one_mask, dim=0)
            min_val, min_ind = torch.min(output + (1 - one_mask), dim=0)

            ext_val, ext_ind = (max_val, max_ind) if (max_val > (1 - min_val)) else (min_val, min_ind)
            g.mask[ext_ind] = torch.tensor(1.0) if ext_val > 0.5 else torch.tensor(0.0)
            # push the current index to Q
            if ext_ind == change_ind:
                g.mask[ext_ind] = 1 - g.mask[ext_ind]
            print('Assign No. ', ext_ind.item(), 'with prob: ', ext_val.item(), 'as value: ', g.mask[ext_ind].item())
            
            l_node_new = []
            for i in l_node:
                if i != ext_ind:
                    l_node_new.append(i)
            l_node = torch.tensor(l_node_new)

        # literal index
        layer_mask = g.forward_level == 0
        l_node = g.forward_index[layer_mask]
        print('Prob: ', output[l_node])
        
        sol = g.mask[l_node]
        # check the satifiability
        sat = pyg_simulation(g, sol)[0]
        print('Solution: ', sol)
        if sat:
            print('====> Hit the correct solution during the backtracking...')
            return sol, sat
        else:
            print('Wrong..')

    return None, 0
    