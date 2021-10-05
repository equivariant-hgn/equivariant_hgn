import networkx as nx
import numpy as np
import scipy
import pickle
import scipy.sparse as sp
import torch
from src.SparseMatrix import SparseMatrix
from src.DataSchema import DataSchema, Entity, Relation, SparseMatrixData
from data_loader_nc import data_loader


DATA_FILE_DIR = '../../data/hgb/NC/'

def load_data(prefix='DBLP', use_node_attrs=True, use_edge_data=True):
    dl = data_loader(DATA_FILE_DIR + prefix)

    # Create Schema
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

    # Collect data
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


    target_entity = 0 #TODO: Check if this is true for all datasets
    n_outputs = dl.nodes['count'][target_entity]
    n_output_classes = dl.labels_train['num_classes']
    schema_out = DataSchema([entities[target_entity]],
                            [Relation(0, 
                                      [entities[target_entity],
                                       entities[target_entity]],
                                       is_set=True)])
    data_target = SparseMatrixData(schema_out)
    data_target[0] = SparseMatrix(
        indices = torch.arange(n_outputs, dtype=torch.int64).repeat(2,1),
        values=torch.zeros([n_outputs, n_output_classes]),
        shape=(n_outputs, n_outputs, n_output_classes),
        is_set=True)
    labels = np.zeros((dl.nodes['count'][0],
                       dl.labels_train['num_classes']), dtype=int)
    val_ratio = 0.2
    train_idx = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_idx)
    split = int(train_idx.shape[0]*val_ratio)
    val_idx = train_idx[:split]
    train_idx = train_idx[split:]
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.nonzero(dl.labels_test['mask'])[0]
    labels[train_idx] = dl.labels_train['data'][train_idx]
    labels[val_idx] = dl.labels_train['data'][val_idx]
    if prefix != 'IMDB':
        labels = labels.argmax(axis=1)
    train_val_test_idx = {}
    train_val_test_idx['train_idx'] = train_idx
    train_val_test_idx['val_idx'] = val_idx
    train_val_test_idx['test_idx'] = test_idx
    return schema,\
           schema_out, \
           data, \
           data_target, \
           labels,\
           train_val_test_idx,\
           dl