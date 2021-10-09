import numpy as np
import torch
import os
from collections import defaultdict
from data_loader_lp import data_loader
from src.DataSchema import DataSchema, Entity, Relation, SparseMatrixData
from src.SparseMatrix import SparseMatrix


DATA_FILE_DIR = '../../data/LP/'

def load_data(prefix='LastFM', use_node_attrs=True, use_edge_data=True):
    dl = data_loader(DATA_FILE_DIR+prefix)

    entities = [Entity(entity_id, n_instances)
                for entity_id, n_instances
                in sorted(dl.nodes['count'].items())]
    relations = [Relation(rel_id, [entities[entity_i], entities[entity_j]])
                    for rel_id, (entity_i, entity_j)
                    in sorted(dl.links['meta'].items())]
    num_relations = len(relations)
    if use_node_attrs:
        # Create fake relations to represent node attributes
        for entity in entities:
            rel_id = num_relations + entity.id
            rel = Relation(rel_id, [entity, entity], is_set=True)
            relations.append(rel)
    schema = DataSchema(entities, relations)

    data = SparseMatrixData(schema)
    for rel_id, data_matrix in dl.links['data'].items():
        # Get subset belonging to entities in relation
        start_i = dl.nodes['shift'][relations[rel_id].entities[0].id]
        end_i = start_i + dl.nodes['count'][relations[rel_id].entities[0].id]
        start_j = dl.nodes['shift'][relations[rel_id].entities[1].id]
        end_j = start_j + dl.nodes['count'][relations[rel_id].entities[1].id]
        rel_matrix = data_matrix[start_i:end_i, start_j:end_j]
        data[rel_id] = SparseMatrix.from_scipy_sparse(rel_matrix.tocoo())
        if not use_edge_data:
            # Use only adjacency information
            data[rel_id].values = torch.ones(data[rel_id].values.shape)

    for ent_id, attr_matrix in dl.nodes['attr'].items():
        if attr_matrix is None:
            # Attribute for each node is a single 1
            attr_matrix = np.ones(dl.nodes['count'][ent_id])[:, None]
        n_channels = attr_matrix.shape[1]
        rel_id = ent_id + num_relations
        n_instances = dl.nodes['count'][ent_id]
        indices = torch.arange(n_instances).unsqueeze(0).repeat(2, 1)
        data[rel_id] = SparseMatrix(
            indices = indices,
            values = torch.FloatTensor(attr_matrix),
            shape = np.array([n_instances, n_instances, n_channels]),
            is_set = True)


    target_rel_id = dl.test_types[0]
    ent_i, ent_j = relations[target_rel_id].entities

    #TODO: this next line might cause an issue
    schema_out = DataSchema([ent_i, ent_j], relations[target_rel_id])

    return schema,\
           schema_out, \
           data, \
           dl

def get_shifts(dl, edge_type):
    ent_i = dl.links['meta'][edge_type][0]
    ent_j = dl.links['meta'][edge_type][1]
    shift_i = dl.nodes['shift'][ent_i]
    shift_j = dl.nodes['shift'][ent_j]   
    return shift_i, shift_j

def get_train_valid_pos(dl, edge_type):
    train_pos, valid_pos = dl.get_train_valid_pos()
    train_pos_arr = np.array(train_pos[edge_type])
    valid_pos_arr = np.array(valid_pos[edge_type])
    shift_i, shift_j = get_shifts(dl, edge_type)
    train_pos_head_full = train_pos_arr[0] - shift_i
    train_pos_tail_full = train_pos_arr[1] - shift_j
    valid_pos_head_full = valid_pos_arr[0] - shift_i
    valid_pos_tail_full = valid_pos_arr[1] - shift_j
    return train_pos_head_full, train_pos_tail_full, \
            valid_pos_head_full, valid_pos_tail_full

def get_train_neg(dl, edge_type=None, edge_types=None):
    if edge_types is None:
        if edge_type is None:
            edge_types = []
        else:
            edge_types = [edge_type]
    train_neg_arr = np.array(dl.get_train_neg(edge_types)[edge_type])
    shift_i, shift_j = get_shifts(dl, edge_type)
    train_neg_head = train_neg_arr[0] - shift_i
    train_neg_tail = train_neg_arr[1] - shift_j
    return train_neg_head, train_neg_tail

def get_valid_neg(dl, edge_type=None, edge_types=None):
    if edge_types is None:
        if edge_type is None:
            edge_types = []
        else:
            edge_types = [edge_type]
    train_neg_arr = np.array(dl.get_valid_neg(edge_types)[edge_type])
    shift_i, shift_j = get_shifts(dl, edge_type)
    valid_neg_head = train_neg_arr[0] - shift_i
    valid_neg_tail = train_neg_arr[1] - shift_j
    return valid_neg_head, valid_neg_tail

def get_valid_neg_2hop(dl, edge_type):
    train_neg_arr = np.array(dl.get_valid_neg_2hop(edge_type))
    shift_i, shift_j = get_shifts(dl, edge_type)
    valid_neg_head = train_neg_arr[0] - shift_i
    valid_neg_tail = train_neg_arr[1] - shift_j
    return valid_neg_head, valid_neg_tail

def get_test_neigh(dl, edge_type=None, neigh_type=None):
    if neigh_type == 'w_random':
        get_test_neigh = dl.get_test_neigh_w_random
    elif neigh_type == 'full_random':
        get_test_neigh = dl.get_test_neigh_full_random
    else:
        get_test_neigh = dl.get_test_neigh
    test_neigh, test_label = get_test_neigh()
    test_neigh_arr = np.array(test_neigh[edge_type])
    test_label_arr = np.array(test_label[edge_type])
    shift_i, shift_j = get_shifts(dl, edge_type)
    test_neigh_head = test_neigh_arr[0] - shift_i
    test_neigh_tail = test_neigh_arr[1] - shift_j
    return test_neigh_head, test_neigh_tail, test_label_arr

def get_test_neigh_from_file(dl, dataset, edge_type):
    save = np.loadtxt(os.path.join(dl.path, f"{dataset}_ini_{edge_type}_label.txt"), dtype=int)
    shift_i, shift_j = get_shifts(dl, edge_type)

    left  = save[0] - shift_i
    right = save[1] - shift_j
    # Don't have access to real labels, just get random
    test_label = np.random.randint(2, size=save[0].shape[0])
    return left, right, test_label



def gen_file_for_evaluate(dl, target_edges, edges, confidence, edge_type, file_path):
    """
    :param edge_list: shape(2, edge_num)
    :param confidence: shape(edge_num,)
    :param edge_type: shape(1)
    :param file_path: string
    """
    # First, turn output into dict
    output_dict = defaultdict(dict)
    shift_l, shift_r = get_shifts(dl, edge_type)
    for l,r,c in zip(edges[0], edges[1], confidence):
            l_i = l + shift_l
            r_i = r + shift_r
            output_dict[l_i][r_i] = c
    # Then, write all the target test edges
    with open(file_path, "a") as f:
        for l,r in zip(target_edges[0], target_edges[1]):
            l_i = l + shift_l
            r_i = r + shift_r
            c = output_dict[l_i][r_i]
            f.write(f"{l_i}\t{r_i}\t{edge_type}\t{c}\n")
