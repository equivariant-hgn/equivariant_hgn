#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import torch
import torch.nn as nn
from src.DataSchema import Data, DataSchema, Relation, SparseMatrixData
from src.SparseMatrix import SparseMatrix


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