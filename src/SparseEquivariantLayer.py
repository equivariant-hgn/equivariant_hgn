#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import torch
import torch.nn as nn
import itertools
import logging
from src.utils import get_all_input_output_partitions, SPARSE_PREFIX_DIMS
from src.DataSchema import Data
from src.SparseTensor import SparseTensor

LOG_LEVEL = logging.ERROR
class SparseEquivariantLayerBlock(nn.Module):
    # Layer mapping between two relations
    def __init__(self, input_dim, output_dim, relation_in, relation_out):
        super(SparseEquivariantLayerBlock, self).__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.input_output_partitions = get_all_input_output_partitions(relation_in, relation_out, SPARSE_PREFIX_DIMS)
        self.n_params = len(self.input_output_partitions)
        stdv = 0.1 / math.sqrt(self.in_dim)
        self.weights = nn.Parameter(torch.Tensor(self.n_params, self.in_dim, self.out_dim).uniform_(-stdv, stdv))
        self.bias = nn.Parameter(torch.ones(self.n_params))
        self.output_shape = [0, self.out_dim] + [entity.n_instances for entity in relation_out.entities]
        self.logger = logging.getLogger()
        self.logger.setLevel(LOG_LEVEL)

    
    def diag(self, X):
        '''
        #Takes diagonal along any input dimensions that are equal,
        #For example, with a a rank 3 tensor X, D = [{0, 2} , {1}] will 
        #produce a rank 2 tensor whose 0th dimension is the 1st dimension of X,
        #and whose 1st dimension is the diagonal of the 0th and 2nd dimensions
        #of X
        '''
        
        # Get dimensions to take diagonals along
        input_diagonals = [sorted(list(i)) for i, o in self.equality_mapping if i != set()]
        list_of_diagonals = input_diagonals.copy()
        # Take diagonal along each dimension
        for i in range(len(list_of_diagonals)):
            diagonal_dims = list_of_diagonals[i]
            # First, permute all dimensions in diagonal_dims to end of tensor
            permutation = list(range(X.ndimension()))
            for dim in sorted(diagonal_dims, reverse=True):
                permutation.append(permutation.pop(dim))
            X = X.permute(permutation)
            # Next, update list_of_diagonals to reflect this permutation

            update_list_of_diagonals = list_of_diagonals.copy()
            for j, input_dims in enumerate(list_of_diagonals[i:]):
                update_list_of_diagonals[i+j] = [permutation.index(input_dim) for input_dim in input_dims]
            list_of_diagonals = update_list_of_diagonals
            # Finally, take a diagonal of the last len(diagonal_dims) dimensions
            n_diagonal_dims = len(diagonal_dims)
            if n_diagonal_dims == 1:
                continue
            else:
                for j in range(n_diagonal_dims - 1):
                    X = X.diagonal(0, -2, -1)

        # Update dimensions in equality_mapping
        updated_equalities = self.equality_mapping.copy()
        for i, diag_dims in enumerate(input_diagonals):
            for j, (input_dims, output_dims) in enumerate(self.equality_mapping):
                if set(input_dims) == set(diag_dims):
                    updated_equalities[j] = ({SPARSE_PREFIX_DIMS+i}, self.equality_mapping[j][1])
        self.equality_mapping = updated_equalities
        return X


    def pool(self, X):
        '''
        Sum along any input dimensions that are not in the output
        Update the equality mappings
        '''
        # Pool over dimensions with no correponding output
        index_array = np.arange(len(X.shape))
        pooling_dims = []
        for i, o in self.equality_mapping:
            if o == set():
                pooling_dims += list(i)
        pooling_dims = sorted(pooling_dims)
        if pooling_dims != []:
            # TODO: can make this a max or mean
            X = X.pool(pooling_dims)

            # Get updated indices
            index_array = np.delete(index_array, pooling_dims)
            update_mapping = {value: index for index, value in enumerate(index_array)}

            # Replace equality mapping with new indices
            new_equality_mapping = []
            for i, o in self.equality_mapping:
                if o == set():
                    continue
                i_new = set()
                for el in i:
                    i_new.add(update_mapping[el])
                new_equality_mapping.append((i_new, o))
            self.equality_mapping = new_equality_mapping

        return X

    def broadcast(self, X, X_out):
        '''
        Expand X to add a new dimension for every output dimension that does 
        not have a corresponding input dimension
        TODO: This is likely much less efficient than it can be
        '''
        n_dimension = X.ndimension()
        matching_in_dims = []
        matching_out_dims  = []
        broadcast_out_dims = []
        broadcast_sizes = []
        n_new_dims = []
        for (i, o) in self.equality_mapping:
            matching_in_dims += list(i)
            matching_out_dims += list(o)[:len(i)]
            broadcast_dims_i = list(o)[len(i):]
            broadcast_out_dims += broadcast_dims_i
            broadcast_sizes += [X_out.shape[i] for i in broadcast_dims_i]
            n_new_dims.append(len(broadcast_dims_i))

        X = X.permute(matching_in_dims)

        matching_out_indices = torch.index_select(X_out.indices, 0, 
                                                  torch.LongTensor(matching_out_dims).to(X_out.indices.device))
        broadcast_out_indices = torch.index_select(X_out.indices, 0,
                                                   torch.LongTensor(broadcast_out_dims).to(X_out.indices.device))

        X = X.broadcast(broadcast_sizes, broadcast_out_indices, matching_out_indices)
        # Update equality mapping
        n_dims_total = n_dimension
        for index, (i, o) in enumerate(self.equality_mapping):
            # Get now-permuted input dimensions
            new_input = set(matching_in_dims[idx] for idx in i)
            # Add in broadcasted dimensions
            new_input |= set(range(n_dims_total,  n_dims_total+n_new_dims[index]))
            n_dims_total += n_new_dims[index]
            self.equality_mapping[index] = (new_input, o)

        return X

    def diag_mask(self, X):
        '''
        #Takes diagonal along any input dimensions that are equal,
        #For example, with a a rank 3 tensor X, D = [{0, 2} , {1}] will
        #produce a rank 2 tensor whose 0th dimension is the 1st dimension of X,
        #and whose 1st dimension is the diagonal of the 0th and 2nd dimensions
        #of X
        '''

        # Get dimensions to take diagonals along
        input_diagonals = [sorted(list(i)) for i, o in self.equality_mapping if i != set() and len(i) > 1]
        # Take diagonal along each dimension
        for diagonal_dims in input_diagonals:
            for dim1, dim2 in zip(diagonal_dims[:-1], diagonal_dims[1:]):
                X = X.diagonal_mask(dim1, dim2)
        return X


    def undiag(self, X, X_out):
        '''
        Expand out any dimensions that are equal in the output tensor 
        to their diagonal
        '''
        for index, (i, o)  in enumerate(self.equality_mapping):

            input_dim = list(i)[0]

            # First, permute input_dim to end of tensor
            permutation = list(range(X.ndimension()))
            permutation.append(permutation.pop(input_dim))
            X = X.permute(permutation)
            # Next, update equality_mapping to reflect this permutation
            for index2, (i2, o2) in enumerate(self.equality_mapping):
                self.equality_mapping[index2] = ({permutation.index(dim) for dim in i2}, o2)

            # Then, embed diag it: expand it out to more dimensions
            n_diagonal_dims = len(o)
            if n_diagonal_dims == 1:
                continue
            else:
                for j in range(n_diagonal_dims - 1):
                    X = X.diag_embed(0, -2, -1)
            # update equality mapping so that input is replaced by last n_dim dimensions of tensor
            new_input = np.arange(X.ndim - n_diagonal_dims, X.ndim)
            self.equality_mapping[index] = ({*new_input}, o)
        return X

    def reindex(self, X):
        '''
        Permute the dimensions of X based on the equality mapping
        '''
        input_dims = []
        output_dims= []
        for i, o in self.equality_mapping:
            input_dims += list(i)
            output_dims += list(o)
        # Sort input_dims by output_dims
        permutation = [x for _, x in sorted(zip(output_dims, input_dims))]
        X = X.permute(permutation)
        return X

    def forward(self, X_in, X_out):
        '''
        X_in: Source sparse tensor
        X_out: Correpsonding sparse tensor for target relation
        '''
        self.logger.info("n_params: {}".format(self.n_params))
        Y = SparseTensor.from_other_sparse_tensor(X_out, self.out_dim)
        for i in range(self.n_params):
            Y  += self.bias[i]

            self.equality_mapping = self.input_output_partitions[i]
            self.logger.info(str(i))

            Y_out = self.diag(X_in)
            if Y_out.nnz() == 0:
                self.logger.info("Diag NNZ = 0")
                continue
            Y_out = self.pool(Y_out)
            if Y_out.nnz() == 0:
                self.logger.info("Pool NNZ = 0")
                continue
            Y_out = self.broadcast(Y_out, X_out)
            if Y_out.nnz() == 0:
                self.logger.info("Broadcast NNZ = 0")
                continue
            Y_out = self.diag_mask(Y_out)
            if Y_out.nnz() == 0:
                self.logger.info("Diag_mask NNZ = 0")
                continue
            Y_out = self.reindex(Y_out)
            
            weight_i = self.weights[i]
            Y_out =  weight_i.T @ Y_out

            Y  += Y_out
        return Y


class SparseEquivariantLayer(nn.Module):
    def __init__(self, schema, input_dim=1, output_dim=1, target_rel=None):
        super(SparseEquivariantLayer, self).__init__()
        self.schema = schema
        self.target_rel = target_rel
        if target_rel == None:
            self.relation_pairs = list(itertools.product(self.schema.relations,
                                                    self.schema.relations))
        else:
            self.relation_pairs = [(rel_i, self.schema.relations[target_rel]) \
                                   for rel_i in self.schema.relations]
        block_modules = {}
        self.input_dim = input_dim
        self.output_dim = output_dim
        for relation_i, relation_j in self.relation_pairs:
            block_module = SparseEquivariantLayerBlock(self.input_dim, self.output_dim,
                                                       relation_i, relation_j)
            block_modules[str((relation_i.id, relation_j.id))] = block_module
        self.block_modules = nn.ModuleDict(block_modules)
        self.logger = logging.getLogger()

    def forward(self, data):
        data_out = Data(self.schema)
        for relation_i, relation_j in self.relation_pairs:
            self.logger.info("Relation: ({}, {})".format(relation_i.id, relation_j.id))
            X_in = data[relation_i.id]
            Y_in = data[relation_j.id]
            layer = self.block_modules[str((relation_i.id, relation_j.id))]
            Y_out = layer.forward(X_in, Y_in)
            if relation_j.id not in data_out:
                data_out[relation_j.id] = Y_out
            else:
                data_out[relation_j.id] = data_out[relation_j.id] + Y_out
        return data_out

