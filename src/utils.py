#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import itertools
import torch


# Non-relation dimensions (batch and channel)
MATRIX_PREFIX_DIMS = 1
SPARSE_PREFIX_DIMS = 0
DENSE_PREFIX_DIMS = 2

# https://stackoverflow.com/questions/19368375/set-partitions-in-python/30134039
def get_partitions(collection):
    if len(collection) == 1:
        yield [ collection ]
        return
    first = collection[0]
    for smaller in get_partitions(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset 
        yield [ [ first ] ] + smaller
    

def get_all_entity_partitions(combined_entities, prefix_dims=0):
    '''
    prefix_dims: offset for dimension numbering
    Returns: List of all mappings between entities and partitioning of their 
    indicies in the concatenated input and output
    '''
    all_entities = np.unique(combined_entities)

    # Map of entity: index
    partitions = {}
    for entity in all_entities:
        entity_indices = prefix_dims + np.where(combined_entities == entity)[0]
        partitions[entity.id] = list(get_partitions(list(entity_indices)))
    
    partition_product = itertools.product(*partitions.values())
    entity_partitions = []
    for combination in partition_product:
        entity_partition_map = {}
        for i, entity in enumerate(all_entities):
            entity_partition_map[entity.id] = combination[i]
        entity_partitions.append(entity_partition_map)
    return entity_partitions

def get_all_input_output_partitions(relation_in, relation_out, prefix_dims=1):
    '''
    prefix_dims: Number of non-relational dimensions (e.g. channel or batch).
                 Used as offset for dimension listings
    Returns: a list of all "input output partitions", which are tuple pairs
    of the set of indices in the input and the set of indices in the output
    that are equal to each other.
    '''
    entities_in = relation_in.entities
    entities_out = relation_out.entities
    combined_entities = np.array(entities_in + entities_out)
    entity_partitions = get_all_entity_partitions(combined_entities, prefix_dims)
    relation_in_length = len(relation_in.entities)

    output = []
    for entity_partition in entity_partitions:
        mapping = []  
        for partitions in entity_partition.values():
            for partition in partitions:
                # get indices with respect to input and output instead of
                # with respect to their concatenation
                inputs = []
                outputs = []
                for entity_index in partition:
                    if entity_index < prefix_dims + relation_in_length:
                        inputs.append(entity_index)
                    else:
                        outputs.append(entity_index - relation_in_length)
                mapping.append((set(inputs), set(outputs)))
        output.append(mapping)
    return output


def update_observed(observed_old, p_keep, min_observed):
    '''
    Updates observed entries
    Source: https://github.com/drgrhm/exch_relational/blob/master/util.py#L186
    Parameters:
        observed_old: old binary array of observed entries
        p_keep: proportion of observed entries to keep
        min_observed: minimum number of entries per row and column
    Returns:
        observed_new: updated array of observed entries
    '''

    inds_sc = np.array(np.nonzero(observed_old)).T

    n_keep = int(p_keep * inds_sc.shape[0])
    n_drop = inds_sc.shape[0] - n_keep

    inds_sc_keep = np.concatenate( (np.ones(n_keep), np.zeros(n_drop)) )
    np.random.shuffle(inds_sc_keep)
    inds_sc = inds_sc[inds_sc_keep == 1, :]

    observed_new = np.zeros_like(observed_old)
    observed_new[inds_sc[:,0], inds_sc[:,1]] = 1

    shape = observed_new.shape
    rows = np.sum(observed_new, axis=1)
    for i in np.array(range(shape[0]))[rows < min_observed]:
        diff = observed_old[i, :] - observed_new[i, :]
        resample_inds = np.array(range(shape[1]))[diff == 1]
        jj = np.random.choice(resample_inds, int(min_observed - rows[i]), replace=False)
        observed_new[i, jj] = 1

    cols = np.sum(observed_new, axis=0)
    for j in np.array(range(shape[1]))[cols < min_observed]:
        diff = observed_old[:, j] - observed_new[:, j]
        resample_inds = np.array(range(shape[0]))[diff == 1]
        ii = np.random.choice(resample_inds, int(min_observed - cols[j]), replace=False)
        observed_new[ii, j] = 1

    return observed_new


def get_ops(partition):
    '''
    p: pool
    g: gather
    i: id
    t: transpose
    b: broadcast
    e: embed
    '''
    input_op = set()
    output_op = set()
    
    for inp, out in partition:
        if out == set():
            if inp == {1, 2}:
                input_op.add("p_diag")
            elif inp == {1} == inp:
                input_op.add("p_row")
            elif inp == {2}:
                input_op.add("p_col")
        else:
            if inp == {1, 2}:
                input_op.add("g_diag")

            if inp == {1} and out == {1}:
                input_op.add("i_row")
                output_op.add("i_row")
            elif inp == {2} and out == {2}:
                input_op.add("i_col")
                output_op.add("i_col")
            elif inp == {1} and out == {2}:
                input_op.add("i_row")
                output_op.add("t_row")
            elif inp == {2} and out == {1}:
                input_op.add("i_col")
                output_op.add("t_col")

        if inp == set():
            if out == {1, 2}:
                output_op.add("b_diag")
            elif out == {1}:
                output_op.add("b_row")
            elif out == {2}:
                output_op.add("b_col")
        else:
            if out == {1,2}:
                output_op.add("e_diag")

    if "p_row" in input_op and "p_col" in input_op:
        input_op = "p_all"
    elif "i_row" in input_op and "i_col" in input_op:
        input_op = "i_all"
    else:
        input_op.discard("i_row")
        input_op.discard("i_col")
        if len(input_op) == 1:
            input_op = input_op.pop()
        elif input_op == set():
            input_op = None
        else:
            raise AssertionError("Can only have single input op")
    if "b_row" in output_op and "b_col" in output_op:
        output_op = "b_all"
    elif "i_row" in output_op and "i_col" in output_op:
        output_op = "i_all"
    elif "t_row" in output_op and "t_col" in output_op:
        output_op = "t_all"
    else:
        output_op.discard("i_row")
        output_op.discard("i_col")
        output_op.discard("t_row")
        output_op.discard("t_col")
        if len(output_op) == 1:
            output_op = output_op.pop()
        elif output_op == set():
            output_op = None
        else:
            raise AssertionError("Can only have single output op")

    return input_op, output_op


def get_all_ops(relation_in, relation_out, prefix_dims=1):
    partitions = get_all_input_output_partitions(relation_in, relation_out, prefix_dims)
    return [get_ops(partition) for partition in partitions]

def get_ops_from_partitions(partitions, input_is_set=False, output_is_set=False):
    all_ops = [get_ops(partition) for partition in partitions]
    if input_is_set:
        all_ops = [(op_in, op_out) for op_in, op_out in all_ops if op_in.split('_')[1] == 'diag']
    if output_is_set:
        all_ops = [(op_in, op_out) for op_in, op_out in all_ops if op_out.split('_')[1] == 'diag']
    return all_ops

def get_masks_of_intersection(array1, array2):
    # Return the mask of values of indices of array2 that intersect with array1
    # For example a1 = [0, 1, 2, 5], a2 = [1, 3, 2, 4], then intersection = [1, 2]
    # and array1_intersection_mask = [False, True, True, False]
    # and array2_intersection_mask = [True, False, True, False]
    n_in = array1.shape[1]
    combined = torch.cat((array1, array2), dim=1)
    intersection, intersection_idx, counts = combined.unique(return_counts=True, return_inverse=True, dim=1)
    intersection_mask = (counts > 1).T[intersection_idx].T
    array1_intersection_mask = intersection_mask[:n_in]
    array2_intersection_mask = intersection_mask[n_in:]
    return array1_intersection_mask, array2_intersection_mask

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)