# -*- coding: utf-8 -*-
import torch
import numpy as np
import pdb
from src.utils import get_masks_of_intersection

class SparseTensor:
    '''
    '''
    def __init__(self, indices, values, shape):
        assert shape.dtype == np.dtype('int64')
        assert len(shape) == indices.shape[0], "Number of dimensions in shape and indices do not match"
        assert indices.shape[1] == values.shape[1], "Number of nonzero elements in indices and values do not match"
        # array of indices with dimensions (n_dimensions, nnz)
        self.indices = indices
        # Array of values with dimensions (n_channels, nnz)
        self.values = values
        # Numpy array of tensor dimension
        self.shape = shape

    @classmethod
    def from_torch_sparse(cls, sparse_tensor):
        '''
        Initialize from pytorch's built-in sparse tensor
        '''
        indices = sparse_tensor.indices()
        values = sparse_tensor.values().T
        shape = np.array(sparse_tensor.shape[:-1])
        return cls(indices, values, shape)

    @classmethod
    def from_dense_tensor(cls, tensor):
        '''
        Initialize from a dense tensor
        '''
        sparse_tensor = tensor.to_sparse()
        return SparseTensor.from_torch_sparse(sparse_tensor)

    @classmethod
    def from_other_sparse_tensor(cls, sparse_tensor, n_channels=None):
        '''
        Initialize all-zero sparse tensor from the indices of another sparse tensor
        If n_channels is specified, use this as the number of channels of the ouput
        '''
        out = sparse_tensor.clone().zero_()
        if n_channels != None:
            out.values = out.values[[0]*n_channels]
        return out

    def ndimension(self):
        return self.indices.shape[0]

    def nnz(self):
        return self.indices.shape[1]
    
    def num_channels(self):
        return self.values.shape[0]

    def size(self, dim=None):
        if dim == None:
            return self.shape
        else:
            return self.shape[dim]

    def to(self, *args, **kwargs):
        self.indices = self.indices.to(*args, **kwargs)
        self.values = self.values.to(*args, **kwargs)
        return self

    def add_sparse_tensor(self, other):
        assert (self.shape == other.shape).all(), "Mismatching shapes"
        assert self.num_channels() == other.num_channels(), "Mismatching number of channels"
        combined_indices = torch.cat((self.indices, other.indices), dim=1)
        combined_values = torch.cat((self.values, other.values), dim=1)
        return SparseTensor(combined_indices, combined_values, self.shape).coalesce()

    def add_tensor(self, other):
        values_out = self.values + other
        return SparseTensor(self.indices, values_out, self.shape)
        
    def add(self, other):
        if type(other) is SparseTensor:
            return self.add_sparse_tensor(other)
        elif type(other) == torch.nn.parameter.Parameter or type(other) == torch.Tensor:
            return self.add_tensor(other)

    def __add__(self, other):
        return self.add(other)
    
    def add_sparse_tensor_(self, other):
        assert (self.shape == other.shape).all(), "Mismatching shapes"
        assert self.num_channels() == other.num_channels(), "Mismatching number of channels"
        combined_indices = torch.cat((self.indices, other.indices), dim=1)
        combined_values = torch.cat((self.values, other.values), dim=1)
        combined_tensor = torch.sparse_coo_tensor(combined_indices, combined_values.T,
                                size=tuple(self.shape) + (self.num_channels(),))
        combined_tensor = combined_tensor.coalesce()
        self.indices = combined_tensor.indices()
        self.values = combined_tensor.values().T
        return self

    def add_tensor_(self, other):
        self.values = self.values + other
        return self
        
    def add_(self, other):
        if type(other) is SparseTensor:
            return self.add_sparse_tensor_(other)
        elif type(other) == torch.nn.parameter.Parameter or type(other) == torch.Tensor:
            return self.add_tensor_(other)

    def __iadd__(self, other):
        return self.add_(other)

    def __matmul__(self, other):
        values_out = self.values @ other
        return SparseTensor(self.indices, values_out, self.shape)
    
    def __rmatmul__(self, other):
        values_out = other @ self.values
        return SparseTensor(self.indices, values_out, self.shape)

    def permute(self, permutation):
        shape_out = self.shape[permutation]
        indices_out = self.indices[permutation]    
        return SparseTensor(indices_out, self.values, shape_out)

    def permute_(self, permutation):
        '''
        In-place version of permute
        '''
        self.shape = self.shape[permutation]
        self.indices = self.indices[permutation]
        return self
    
    def transpose(self, dim1, dim2):
        indices_out = self.indices.clone()
        shape_out = self.shape.copy()

        tmp = indices_out[dim1].clone()
        tmp_dim = shape_out[dim1].copy()

        indices_out[dim1] = indices_out[dim2]
        shape_out[dim1] = shape_out[dim2]

        indices_out[dim2] = tmp
        shape_out[dim2] = tmp_dim
        return SparseTensor(indices_out, self.values, shape_out)

    def diagonal(self, offset=0, dim1=0, dim2=1):
        if dim1 < 0:
            dim1 = self.ndimension() + dim1
        if dim2 < 0:
            dim2 = self.ndimension() + dim2
        '''
        Returns a partial view of input with the its diagonal elements with
        respect to dim1 and dim2 appended as a dimension at the end of the shape.
        Requires dim1 < dim2
        
        offset is a useless parameter to match the method signature of a normal tensor
        '''
        assert dim1 < dim2, "Requires dim1 < dim2"
        assert self.shape[dim1] == self.shape[dim2], "dim1 and dim2 are not of equal length"

        # Get only values and indices where dim1 == dim2from_
        diag_idx = torch.where(self.indices[dim1] == self.indices[dim2])[0]
        diag_values = self.values[:, diag_idx]
        indices_out = torch.index_select(self.indices, 1, diag_idx)

        # Remove diagonal dimensions and append to end
        reshape_indices = np.arange(self.ndimension() + 1)
        reshape_indices = reshape_indices[reshape_indices != dim1]
        reshape_indices = reshape_indices[reshape_indices != dim2]
        reshape_indices[-1] = dim1
        
        return SparseTensor(indices_out, diag_values, self.shape).permute_(reshape_indices)

    def diagonal_mask(self, dim1=0, dim2=1):
        '''
        Zero out all values not on the diagonal of dimensions dim1 and dim2
        '''
        assert self.shape[dim1] == self.shape[dim2], "dim1 and dim2 are not of equal length"
        # Get only values and indices where dim1 == dim2
        diag_idx = torch.where(self.indices[dim1] == self.indices[dim2])[0]
        diag_values = self.values[:, diag_idx]
        indices_out = torch.index_select(self.indices, 1, diag_idx)
        return SparseTensor(indices_out, diag_values, self.shape)
    
    def coalesce_(self):
        '''
        Add all duplicated entries together
        '''
        torch_tensor = torch.sparse_coo_tensor(self.indices, self.values.T,
                                               size=tuple(self.shape) + (self.num_channels(),))
        coalesced_tensor = torch_tensor.coalesce()
        self.indices = coalesced_tensor.indices()
        self.values = coalesced_tensor.values().T
        return self
        
    def coalesce(self):
        '''
        Add all duplicated entries together
        '''
        torch_tensor = torch.sparse_coo_tensor(self.indices, self.values.T,
                                               size=tuple(self.shape) + (self.num_channels(),))
        coalesced_tensor = torch_tensor.coalesce()

        return SparseTensor(coalesced_tensor.indices(), coalesced_tensor.values().T, self.shape)
                                               
        
    def pool(self, pooling_dims):
        remaining_dims = self.remaining_dimensions(pooling_dims)
        pooled_shape = self.shape[remaining_dims]
        pooling_indices = self.indices[remaining_dims, :]
        pooled_tensor = torch.sparse_coo_tensor(pooling_indices, self.values.T,
                                                size=tuple(pooled_shape) + (self.num_channels(),)).coalesce()
        pooled_indices = pooled_tensor.indices()
        pooled_values = pooled_tensor.values().T

        return SparseTensor(pooled_indices, pooled_values, pooled_shape)


    def get_intersection(self, indices_out_matching):
        '''
        Given sparse pooled input tensor and sparse output tensor, reduce the
        tensor to only indices also found in the output
        '''
        pass
        
    def broadcast(self, new_dim_sizes, indices_out_broadcast, indices_out_matching):
        '''
        Assume self is already coalesced and that indices are sorted
        Add new dimension to end and expand by new_dim
        Use sparsity of intersection of indices and indices_out_matching, and 
        sparsity of indices_out_broadcast for the new dimension
        
        indices_out_matching: array of indices of target with corresponding indices
                              in self, ordered the same as indices of self
        indices_out_broadcast: array of indices of target that do not correspond to self
        new_dim_sizes: array of sizes of the new dims added by indices_out_broadcast
        
        TODO: some of these operations are very redundant because intersection could be dense
        '''
        assert len(new_dim_sizes) == indices_out_broadcast.shape[0], "new_dim_sizes must have sizes for each new broadcast dim"
        assert indices_out_matching.shape[0] == self.ndimension(), "indices_out_matching must be for each dimension in self"

        if indices_out_matching.shape[0] == 0:
            indices_out = indices_out_broadcast
            nnz = indices_out_broadcast.shape[1]
            values_out = self.values.repeat((1, nnz))
        else:
            # Get unique values of matching output indices, as well as a way to reverse this process
            # TODO: can probably use unique_consecutive if indices are drawn in order
            indices_out_m_unique, indices_out_m_inverse_idx = indices_out_matching.unique(dim=1, return_inverse=True)
            
            # Get masks of which values to take from self.indices and from the unique matching output indices
            indices_intersection_mask, indices_out_m_unique_intersection_mask = get_masks_of_intersection(self.indices, indices_out_m_unique)
    
            # Get mask of which values to take from the matching output indices by undoing uniqueness operation
            indices_out_m_intersection_mask = indices_out_m_unique_intersection_mask[indices_out_m_inverse_idx]
    
            # Get matching output indices
            indices_out_m_intersection = indices_out_matching.T[indices_out_m_intersection_mask].T
        
            # Get corresponding indices of dimension to broadcast to
            indices_out_broadcast = indices_out_broadcast.T[indices_out_m_intersection_mask].T

            indices_out = torch.cat([indices_out_m_intersection, indices_out_broadcast])                    
        
            # Get values to take, repeating values to broadcast
            values_intersection = self.values[:, indices_intersection_mask]
            _, value_counts = torch.unique(indices_out_m_intersection, dim=1, return_counts=True)
            values_out = torch.repeat_interleave(values_intersection, value_counts, dim=1)

        assert values_out.shape[1] == indices_out.shape[1], "Output values and indices counts do not match"
        # Get new shape:
        if new_dim_sizes != []:
            shape_out = np.concatenate((self.shape, new_dim_sizes))
        else:
            shape_out = self.shape
        return SparseTensor(indices_out, values_out, shape_out)
    
    def diag_embed(self, indices_tgt):
        '''
        indices_tgt: (2, nnz) target indices of dimensions to embed diagonal onto
        Embed the source tensor onto two highest dimensions
        '''
        indices_out = self.indices[-1]
        torch.cat(indices)
        torch.cat(indices_out)
        #TODO: complete this
        return SparseTensor(indices_out, self.values, self.shape)

    def remaining_dimensions(self, dims):
        '''
        Given list of dimensions, return all dimensions but those ones
        e.g. for 5-dimensional tensor, removing [1, 3] results in [0,2,4]
        '''
        return sorted(list(set(range(self.ndimension())) - set(dims)))
        
    def clone(self):
        return SparseTensor(self.indices.clone(), self.values.clone(), self.shape.copy())

    def zero_(self):
        self.values = self.values.zero_()
        return self

    def to_torch_sparse(self):
        return torch.sparse_coo_tensor(self.indices, self.values.T,
                                   size=tuple(self.shape) + (self.num_channels(),))

    def to_dense(self):
        out = self.to_torch_sparse().to_dense()
        # Move channel dimension to beginning
        permute_channel_dim = [-1] + list(range(self.ndimension()))
        return out.permute(*permute_channel_dim)