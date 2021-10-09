#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.DataSchema import Data, DataSchema, Relation, SparseMatrixData
from src.utils import DENSE_PREFIX_DIMS
from src.SparseTensor import SparseTensor
from src.SparseMatrix import SparseMatrix
import pdb


class SparseGroupNorm(nn.Module):
    '''
    Normalize each channel separately
    '''
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True):
        super(SparseGroupNorm, self).__init__()
        self.eps = eps
        self.num_groups = num_groups
        self.num_channels = num_channels
        assert self.num_groups == self.num_channels, "Currently only implemented for num_groups == num_channels"
        self.gamma = nn.Parameter(torch.ones(self.num_channels))
        self.affine = affine
        if self.affine:
            self.beta = nn.Parameter(torch.zeros(self.num_channels))
    
    def forward(self, sparse_tensor):
        values = sparse_tensor.values
        var, mean = torch.var_mean(values, dim=1, unbiased=False)
        values_out = (self.gamma * (values.T - mean) / torch.sqrt(var + self.eps)).T
        if self.affine:
            values_out += self.beta
        out = sparse_tensor.clone()
        out.values = values_out
        return out

class SparseMatrixGroupNorm(nn.Module):
    '''
    Normalize each channel separately
    '''
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True):
        super(SparseMatrixGroupNorm, self).__init__()
        self.eps = eps
        self.num_groups = num_groups
        self.num_channels = num_channels
        assert self.num_groups == self.num_channels, "Currently only implemented for num_groups == num_channels"
        self.gamma = nn.Parameter(torch.ones(self.num_channels))
        self.affine = affine
        if self.affine:
            self.beta = nn.Parameter(torch.zeros(self.num_channels))
    
    def forward(self, sparse_matrix):
        values = sparse_matrix.values
        var, mean = torch.var_mean(values, dim=0, unbiased=False)
        values_out = (self.gamma * (values - mean) / torch.sqrt(var + self.eps))
        if self.affine:
            values_out += self.beta
        out = sparse_matrix.clone()
        out.values = values_out
        return out

class SparseMatrixLinear(nn.Module):
    '''
    Linear layer applied to a single sparse matrix
    '''
    def __init__(self, *args):
        super(SparseMatrixLinear, self).__init__()
        self.linear = nn.Linear(*args)

    def forward(self, matrix):
        values_out = self.linear(matrix.values)
        shape_out = (matrix.n, matrix.m, values_out.shape[1])
        return SparseMatrix(indices=matrix.indices, values=values_out,
                            shape=shape_out, indices_diag=matrix.indices_diag,
                            is_set=matrix.is_set)

class SparseMatrixRelationLinear(nn.Module):
    '''
    Apply linear layer to sparse matrix for each relation.
    Bring each attribute layer to the same vector space
    Optionally, only apply to node attributes
    '''
    def __init__(self, schema, in_dims, out_dim, node_only=False):
        super(SparseMatrixRelationLinear, self).__init__()
        self.schema = schema
        self.node_only = node_only
        self.linear = nn.ModuleDict()
        if type(in_dims) == int:
            in_dims_dict = {}
            for rel in self.schema.relations:
                in_dims_dict[rel.id] = in_dims
            in_dims = in_dims_dict
        for rel in self.schema.relations:
            linear = SparseMatrixLinear(in_dims[rel.id], out_dim)
            self.linear[str(rel.id)] = linear

    def forward(self, X):
        X_out = SparseMatrixData(X.schema)
        for rel in self.schema.relations:
            if self.node_only and rel.is_set:
                X_out[rel.id] = X[rel.id]
            else:
                X_out[rel.id] = self.linear[str(rel.id)](X[rel.id])
        return X_out

class RelationNorm(nn.Module):
    '''
    Normalize each channel of each relation separately
    '''
    def __init__(self, schema, num_channels, affine, sparse=False, matrix=False):
        super(RelationNorm, self).__init__()
        self.schema = schema
        self.rel_norms = nn.ModuleDict()
        self.sparse = True
        if sparse:
            if matrix:
                GroupNorm = SparseMatrixGroupNorm
            else:
                GroupNorm = SparseGroupNorm
        else:
            GroupNorm = nn.GroupNorm

        for relation in schema.relations:
            rel_norm = GroupNorm(num_channels, num_channels, affine=affine)
            self.rel_norms[str(relation.id)] = rel_norm

    def forward(self, X):
        for relation in self.schema.relations:
            rel_norm = self.rel_norms[str(relation.id)]
            X[relation.id] = rel_norm(X[relation.id])
        return X


class Activation(nn.Module):
    '''
    Extend torch.nn.module modules to be applied to each relation for a
    sparse matrix or tensor.
    Activation can either be an initialized nn.Module, an nn.functional,
    or a ModuleDict of initialized str(relation.id) to nn.Modules.
    If is_sparse=True, then using sparse tensor or matrix, so call .values first
    '''
    def __init__(self, schema, activation, is_dict=False, is_sparse=False):
        super(Activation, self).__init__()
        self.schema = schema
        self.activation = activation
        self.is_dict = is_dict
        self.is_sparse = is_sparse

    def forward(self, X):
        for relation in self.schema.relations:
            if relation.id in X:
                if self.is_dict:
                    activation = self.activation[str(relation.id)]
                else:
                    activation = self.activation
                if self.is_sparse:
                    X[relation.id].values = activation(X[relation.id].values)
                else:
                    X[relation.id] = activation(X[relation.id])
        return X

def functional(function, schema, X, *args, **kwargs):
    '''
    Extend torch.nn.functional functions to be applied to each relation
    '''
    for relation in schema.relations:
        X[relation.id] = function(X[relation.id], *args, **kwargs)
    return X


class ReLU(nn.Module):
    def __init__(self, schema):
        super(ReLU, self).__init__()
        self.schema = schema
    
    def forward(self, X):
        for relation in self.schema.relations:
            X[relation.id] = F.relu(X[relation.id]) 
        return X


class EntityBroadcasting(nn.Module):
    '''
    Given encodings for each entity, returns tensors in the shape of the 
    given relationships in the schema
    '''
    def __init__(self, schema, dims):
        super(EntityBroadcasting, self).__init__()
        self.schema = schema
        self.dims = dims
        # empty param just for holding device info
        self.device_param = nn.Parameter(torch.Tensor(0))

    def make_relation(self, encodings, relation):
        batch_size = encodings.batch_size
        relation_shape = [batch_size, self.dims] + relation.get_shape()
        relation_out = torch.zeros(relation_shape, device=self.device_param.device)
        num_new_dims = len(relation.entities) -1
        for entity_idx, entity in enumerate(relation.entities):
            entity_enc = encodings[entity.id]
            # Expand tensor to appropriate number of dimensions
            for _ in range(num_new_dims):
                entity_enc = entity_enc.unsqueeze(-1)
            # Transpose the entity to the appropriate dimension
            entity_enc.transpose_(DENSE_PREFIX_DIMS, entity_idx+DENSE_PREFIX_DIMS)
            # Broadcast-add to output
            relation_out += entity_enc
        return relation_out

    def forward(self, encodings):
        data_out = Data(self.schema)
        for relation in self.schema.relations:
            data_out[relation.id] = self.make_relation(encodings, relation)
        return data_out

class EntityPooling(nn.Module):
    '''
    Produce encodings for every instance of every entity
    Encodings for each entity are produced by summing over each relation which
    contains the entity
    '''
    def __init__(self, schema, dims):
        super(EntityPooling, self).__init__()
        self.schema = schema
        self.dims = dims
        self.out_shape = [e.n_instances for e in self.schema.entities]
        # Make a "schema" for the encodings
        enc_relations = [Relation(i, [self.schema.entities[i]])
                            for i in range(len(self.schema.entities))]
        self.enc_schema = DataSchema(self.schema.entities, enc_relations)
    
    def get_pooling_dims(self, entity, relation):
        pooling_dims = []
        for entity_dim, rel_entity in enumerate(relation.entities):
            if entity != rel_entity:
                pooling_dims += [DENSE_PREFIX_DIMS + entity_dim]
        return pooling_dims
    
    def pool_tensor(self, X, pooling_dims):
        if len(pooling_dims) > 0:
            return torch.sum(X, pooling_dims)
        else:
            return X

    def pool_tensor_diag(self, X):
        while X.ndim > DENSE_PREFIX_DIMS+1:
            assert X.shape[-1] == X.shape[-2]
            X = X.diagonal(0, -1, -2)
        return X

    def forward(self, data):
        out = Data(self.enc_schema, batch_size=data.batch_size)
        for entity in self.schema.entities:
            for relation in self.schema.relations:
                if entity not in relation.entities:
                    continue
                else:
                    pooling_dims = self.get_pooling_dims(entity, relation)
                    data_rel = data[relation.id]
                    entity_out = self.pool_tensor(data_rel, pooling_dims)
                    entity_out = self.pool_tensor_diag(entity_out)
            out[entity.id] = entity_out
        return out


class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        assert p >= 0 and p < 1.0, "Require 0 <= p < 1"
        self.p = p

    def forward(self, X):
        if self.training:
            binomial = torch.distributions.bernoulli.Bernoulli(probs=1-self.p)
            return X * binomial.sample(X.shape[1:]).to(X.device) * (1.0/(1-self.p))
        return X