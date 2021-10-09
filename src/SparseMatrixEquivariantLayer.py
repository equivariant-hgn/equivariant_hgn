#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import itertools
import logging
from src.utils import get_all_input_output_partitions, \
                        get_ops_from_partitions, MATRIX_PREFIX_DIMS
from src.DataSchema import SparseMatrixData, DataSchema, Relation
from src.SparseMatrix import SparseMatrix

LOG_LEVEL = logging.INFO

class SparseMatrixEquivariantLayerBlock(nn.Module):
    # Layer mapping between two relations
    def __init__(self, input_dim, output_dim, relation_in, relation_out, pool_op='mean'):
        super(SparseMatrixEquivariantLayerBlock, self).__init__()
        assert len(relation_in.entities) == 2, "Relation must be second or first order"
        assert len(relation_out.entities) <= 2, "Relation must be second order"
        in_is_set = relation_in.is_set
        out_is_set = relation_out.is_set
        self.block_id = (relation_in.id, relation_out.id)
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.input_output_partitions = get_all_input_output_partitions(relation_in, relation_out, MATRIX_PREFIX_DIMS)
        self.all_ops = get_ops_from_partitions(self.input_output_partitions,
                                               in_is_set, out_is_set)
        self.n_params = len(self.all_ops)
        stdv = 1. / math.sqrt(self.in_dim)
        self.weights = nn.Parameter(torch.Tensor(self.n_params, self.in_dim, self.out_dim).uniform_(-stdv, stdv))
        self.output_shape = [0, self.out_dim] + [entity.n_instances for entity in relation_out.entities]
        self.logger = logging.getLogger()
        self.logger.setLevel(LOG_LEVEL)
        self.pool_op = pool_op

    def output_op(self, op_str, X_out, data, device):
        op, index_str = op_str.split('_')
        if op == 'b':
            return X_out.broadcast(data, index_str, device)
        elif op == 'e':
            return X_out.embed_diag(data, device)

    def input_op(self, op_str, X_in, device):
        op, index_str = op_str.split('_')
        if op == 'g':
            return X_in.gather_diag(device)
        elif op == 'p':
            return X_in.pool(index_str, device=device, op=self.pool_op)

    def forward(self, X_in, X_out, indices_identity, indices_trans):
        '''
        X_in: Source sparse tensor
        X_out: Correpsonding sparse tensor for target relation
        '''
        self.logger.info("n_params: {}".format(self.n_params))
        if type(X_out) == SparseMatrix:
            Y = SparseMatrix.from_other_sparse_matrix(X_out, self.out_dim)
        else:
            Y = X_out.clone()
        #TODO: can add a cache for input operations here
        for i in range(self.n_params):
            op_inp, op_out = self.all_ops[i]
            weight = self.weights[i]
            device = weight.device
            if op_inp == None:
                X_mul = torch.matmul(X_in, weight)
                X_op_out = self.output_op(op_out, X_out, X_mul, device)
            elif op_out == None:
                X_op_inp = self.input_op(op_inp, X_in, device)
                X_mul = torch.matmul(X_op_inp, weight)
                X_op_out = X_mul
            elif op_out[0] == "i":
                # Identity
                X_intersection_vals = X_in.gather_mask(indices_identity[0])
                X_mul = X_intersection_vals @ weight
                X_op_out = X_out.broadcast_from_mask(X_mul, indices_identity[1], device)
            elif op_out[0] == "t":
                # Transpose
                X_T_intersection_vals = X_in.gather_transpose(indices_trans[0])
                X_mul = X_T_intersection_vals @ weight
                X_op_out = X_out.broadcast_from_mask(X_mul, indices_trans[1], device)
            else:
                # Pool or Gather or Do Nothing
                X_op_inp = self.input_op(op_inp, X_in, device)
                # Multiply values by weight
                X_mul = torch.matmul(X_op_inp, weight)
                # Broadcast or Embed Diag or Transpose
                X_op_out = self.output_op(op_out, X_out, X_mul, device)
            #assert X_op_out.nnz() == X_out.nnz()
            #assert Y.nnz() == X_out.nnz(), "Y: {}, X_out: {}".format(Y.nnz(), X_out.nnz())
            #assert Y.nnz() == X_op_out.nnz(), "Y: {}, X_op_out: {}".format(Y.nnz(), X_op_out.nnz())
            Y = Y + X_op_out
        return Y


class SparseMatrixEquivariantLayer(nn.Module):
    def __init__(self, schema, input_dim=1, output_dim=1, schema_out=None, pool_op='mean'):
        '''
        input_dim: either a rel_id: dimension dict, or an integer for all relations
        output_dim: either a rel_id: dimension dict, or an integer for all relations
        '''
        super(SparseMatrixEquivariantLayer, self).__init__()
        self.schema = schema
        self.schema_out = schema_out
        if self.schema_out == None:
            self.schema_out = schema
        self.relation_pairs = list(itertools.product(self.schema.relations,
                                                    self.schema_out.relations))
        block_modules = {}
        if type(input_dim) == dict:
            self.input_dim = input_dim
        else:
            self.input_dim = {rel.id: input_dim for rel in self.schema.relations}
        if type(output_dim) == dict:
            self.output_dim = output_dim
        else:
            self.output_dim = {rel.id: output_dim for rel in self.schema_out.relations}
        for relation_i, relation_j in self.relation_pairs:
            block_module = SparseMatrixEquivariantLayerBlock(self.input_dim[relation_i.id],
                                                             self.output_dim[relation_j.id],
                                                              relation_i, relation_j,
                                                              pool_op=pool_op)
            block_modules[str((relation_i.id, relation_j.id))] = block_module
        self.block_modules = nn.ModuleDict(block_modules)
        self.logger = logging.getLogger()
        
        bias = {}
        for relation in self.schema_out.relations:
            if relation.entities[0] == relation.entities[1] and not relation.is_set:
                # Square matrix: One bias for diagonal, one for all
                bias[str(relation.id)] = nn.Parameter(torch.zeros(2, self.output_dim[relation.id]))
            else:
                # One bias parameter for full matrix
                bias[str(relation.id)] = nn.Parameter(torch.zeros(1, self.output_dim[relation.id]))
        self.bias = nn.ParameterDict(bias)

    def multiply_matrices(self, data, data_out, indices_identity=None, indices_transpose=None):
        for relation_i, relation_j in self.relation_pairs:
            X_in = data[relation_i.id]
            Y_in = data[relation_j.id]
            layer = self.block_modules[str((relation_i.id, relation_j.id))]
            indices_id = indices_identity[relation_i.id, relation_j.id]
            indices_trans = indices_transpose[relation_i.id, relation_j.id]
            Y_out = layer.forward(X_in, Y_in, indices_id, indices_trans)
            if relation_j.id not in data_out:
                data_out[relation_j.id] = Y_out
            else:
                data_out[relation_j.id] = data_out[relation_j.id] + Y_out
        return data_out
        
    def add_bias(self, data):
        for relation in self.schema_out.relations:
            bias_param = self.bias[str(relation.id)]
            data[relation.id] = data[relation.id] + bias_param[0]
            if relation.entities[0] == relation.entities[1] and not relation.is_set:
                device = bias_param.device
                bias_diag = data[relation.id].broadcast(bias_param[1], 'diag', device)
                data[relation.id] = data[relation.id] + bias_diag
        return data

    def forward(self, data, indices_identity=None, indices_transpose=None):
        data_out = SparseMatrixData(self.schema_out)
        data_out = self.multiply_matrices(data, data_out,
                                          indices_identity, indices_transpose)
        data_out = self.add_bias(data_out)
        return data_out


class SparseMatrixEntityPoolingLayer(SparseMatrixEquivariantLayer):
    '''
    Pool all relations an entity takes part in to create entity-specific embeddings
    '''
    def __init__(self, schema, input_dim=1, output_dim=1, entities=None, pool_op='mean'):
        '''
        input_dim: either a rel_id: dimension dict, or an integer for all relations
        output_dim: either a rel_id: dimension dict, or an integer for all relations
        '''
        if entities == None:
            entities = schema.entities
        enc_relations = [Relation(i, [entity, entity], is_set=True)
                                for i, entity in enumerate(entities)]
        encodings_schema = DataSchema(entities, enc_relations)
        super().__init__(schema, input_dim, output_dim,
                         schema_out=encodings_schema, pool_op=pool_op)

    def multiply_matrices(self, data, data_out, data_target):
        for relation_i, relation_j in self.relation_pairs:
            X_in = data[relation_i.id]
            Y_in = data_target[relation_j.id]
            layer = self.block_modules[str((relation_i.id, relation_j.id))]
            Y_out = layer.forward(X_in, Y_in, None, None)
            if relation_j.id not in data_out:
                data_out[relation_j.id] = Y_out
            else:
                data_out[relation_j.id] = data_out[relation_j.id] + Y_out
        return data_out

    def forward(self, data, data_target):
        data_out = SparseMatrixData(self.schema_out)
        data_out = self.multiply_matrices(data, data_out, data_target)
        data_out = self.add_bias(data_out)
        return data_out

class SparseMatrixEntityBroadcastingLayer(SparseMatrixEquivariantLayer):
    '''
    Given entity-specific embeddings, return activations for the specified
    indices for every relation in the given schema
    '''
    def __init__(self, schema, input_dim=1, output_dim=1, entities=None, pool_op='mean'):
        '''
        schema: schema to broadcast to
        input_dim: either a rel_id: dimension dict, or an integer for all relations
        output_dim: either a rel_id: dimension dict, or an integer for all relations
        entities: if specified, these are the input entities for the encodings
        '''
        if entities == None:
            entities = schema.entities
        enc_relations = [Relation(i, [entity, entity], is_set=True)
                                for i, entity in enumerate(entities)]
        encodings_schema = DataSchema(entities, enc_relations)
        super().__init__(encodings_schema, input_dim, output_dim,
                         schema_out=schema, pool_op=pool_op)

    def multiply_matrices(self, data, data_out, data_target):
        for relation_i, relation_j in self.relation_pairs:
            X_in = data[relation_i.id]
            Y_in = data_target[relation_j.id]
            layer = self.block_modules[str((relation_i.id, relation_j.id))]
            Y_out = layer.forward(X_in, Y_in, None, None)
            if relation_j.id not in data_out:
                data_out[relation_j.id] = Y_out
            else:
                data_out[relation_j.id] = data_out[relation_j.id] + Y_out
        return data_out

    def forward(self, data, data_target=None):
        data_out = SparseMatrixData(self.schema_out)
        data_out = self.multiply_matrices(data, data_out, data_target)
        data_out = self.add_bias(data_out)
        return data_out
