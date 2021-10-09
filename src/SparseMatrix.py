# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch_sparse
import torch_scatter
from src.utils import get_masks_of_intersection, MATRIX_PREFIX_DIMS

class SparseMatrix:
    def __init__(self, indices, values, shape, indices_diag=None, is_set=False):
        assert indices.shape[0] == 2, "Indices must be two dimensions"
        assert indices.shape[1] == values.shape[0], "Number of nonzero elements in indices and values do not match"
        assert len(shape) == 3, "Shape must be 3 dimensions: n, m, and channels"
        assert len(values.shape) == 2, "Currently only support vectors as values"
        assert shape[2] == values.shape[1], "Values channels and shape channels must match"
        assert indices[0].max() <= shape[0]
        assert indices[1].max() <= shape[1]
        # array of indices with dimensions (n_dimensions, nnz)
        self.indices = indices
        # Array of values with dimensions (n_channels, nnz)
        self.values = values

        self.n_channels = shape[2]
        # First and second dimension
        self.n = shape[0]
        self.m = shape[1]
        # If square matrix, initialize indices of transpose and diagonals if not provided
        self.indices_diag = indices_diag

        if self.n == self.m and self.indices_diag == None:
            self.indices_diag = self.calc_indices_diag()
        #assert self.is_sorted()
        self.is_set = is_set

    @classmethod
    def from_torch_sparse(cls, sparse_tensor):
        '''
        Initialize from pytorch's built-in sparse tensor
        '''
        sparse_tensor = sparse_tensor.coalesce()
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        shape = np.array(sparse_tensor.shape)
        return cls(indices, values, shape)

    @classmethod
    def from_scipy_sparse(cls, sparse_matrix):
        values = sparse_matrix.data
        indices = np.vstack((sparse_matrix.row, sparse_matrix.col))

        torch_indices = torch.LongTensor(indices)
        torch_values = torch.FloatTensor(values)
        if torch_values.ndim == 1:
            torch_values = torch_values.unsqueeze(1)
        shape = sparse_matrix.shape + (torch_values.shape[1], )

        torch_sparse =  torch.sparse.FloatTensor(torch_indices, torch_values,
                                                 torch.Size(shape))
        return SparseMatrix.from_torch_sparse(torch_sparse)

    @classmethod
    def from_dense_tensor(cls, tensor, prefix_dims=MATRIX_PREFIX_DIMS):
        '''
        Initialize from a dense tensor
        '''
        # Permute dense dimensions to beginning
        permutation = list(range(prefix_dims, tensor.ndim)) + list(range(prefix_dims))
        sparse_tensor = tensor.permute(*permutation).to_sparse(tensor.ndim - prefix_dims).coalesce()
        return SparseMatrix.from_torch_sparse(sparse_tensor)

    @classmethod
    def from_other_sparse_matrix(cls, sparse_matrix, n_channels=None):
        '''
        Initialize all-zero sparse tensor from the indices of another sparse tensor
        If n_channels is specified, use this as the number of channels of the ouput
        '''
        out = sparse_matrix.clone().zero_()
        if n_channels != None:
            n_values = out.values.shape[0]
            out.values = out.values[:,0].view(n_values, 1).expand(n_values, n_channels)
            out.n_channels = n_channels
        return out

    @classmethod
    def from_embed_diag(cls, values, is_set=False):
        n_instances = values.shape[0]
        n_channels = values.shape[1]
        shape = np.array([n_instances, n_instances, n_channels])
        indices = torch.arange(n_instances).repeat(2,1)
        return SparseMatrix(indices=indices, values=values, shape=shape, is_set=is_set)

    def ndimension(self):
        return self.indices.shape[0]

    def nnz(self):
        assert self.values.shape[0] == self.indices.shape[1],  "Number of nonzero elements in indices and values do not match"
        return self.values.shape[0]

    def size(self, dim=None):
        if dim == None:
            return self.n, self.m, self.n_channels
        else:
            if dim == 0:
                return self.n
            elif dim == 1:
                return self.m
            elif dim == 2:
                return self.n_channels

    def shape(self):
        return self.size()

    def sort_indices(self):
        indices_and_values = torch.cat((self.indices, self.values.T), 0)
        indices_and_values_sorted = torch.unique(indices_and_values, sorted=True)
        indices_out = indices_and_values_sorted[[0,1],:]
        values_out = indices_and_values_sorted[2:, :]
        shape_out = (self.m, self.n, self.n_channels)
        return SparseMatrix(indices_out, values_out, shape_out, is_set=self.is_set)

    def calc_indices_diag(self):
        '''
        Get all indices of values that are along the diagonal
        '''
        return torch.where(self.indices[0] == self.indices[1])[0]

    def calc_intersection_mask(self, other):
        '''
        Given sparse matrix other, return array masks of which indices appear in both
        Assume both self and other have sorted indices
        '''
        return get_masks_of_intersection(self.indices, other.indices)

    def calc_transpose_intersection_overlap(self, other):
        return self.transpose().calc_intersection_mask(other)

    def calc_indices_transpose(self, other):
        '''
        Take other sparse matrix and get indices of transpose
        
        '''
        indices_in = self.indices
        indices_out = self.indices[[1,0], :]
        
        # Get masks of which values to take from original indices and from the output indices
        in_intersection_mask, out_intersection_mask = get_masks_of_intersection(indices_in, indices_out)


        # Get indices of overlap
        indices_in_intersection = indices_in.T[in_intersection_mask].T

        # Get values to take
        values_intersection = self.values[:, indices_in_intersection]

        # Can take these values then add them to target sparse matrix and coalesce
        return SparseMatrix(values_intersection, indices_in_intersection,
                            self.shape(), is_set=self.is_set)

    def to(self, *args, **kwargs):
        self.indices = self.indices.to(*args, **kwargs)
        self.values = self.values.to(*args, **kwargs)
        if self.indices_diag != None:
            self.indices_diag = self.indices_diag.to(*args, **kwargs)
        return self

    def add_sparse_matrix(self, other):
        assert (self.shape() == other.shape()), \
            "Mismatching shapes, self: {} and other: {}".format(self.shape(), other.shape())
        assert self.n_channels == other.n_channels, \
            "Mismatching number of channels, self: {} and other: {}".format(self.n_channels, other.n_channels)
        if (self.nnz() == other.nnz()) and (self.indices == other.indices).all():
            return SparseMatrix(self.indices, self.values + other.values, self.shape(), self.indices_diag, self.is_set)
        else:
            combined_indices = torch.cat((self.indices, other.indices), dim=1)
            combined_values = torch.cat((self.values, other.values), dim=0)
            return SparseMatrix(combined_indices, combined_values, self.shape(), is_set=self.is_set).coalesce()

    def add_tensor(self, other):
        values_out = self.values + other
        return SparseMatrix(self.indices, values_out, self.shape(), self.indices_diag, self.is_set)

    def equal(self, other):
        indices_equal = torch.equal(self.indices, other.indices)
        values_equal = torch.equal(self.values, other.values)
        n_equal = self.n == other.n
        m_equal = self.m == other.m
        return indices_equal and values_equal and n_equal and m_equal
        
    def add(self, other):
        if type(other) is SparseMatrix:
            return self.add_sparse_matrix(other)
        elif type(other) == torch.nn.parameter.Parameter or type(other) == torch.Tensor:
            return self.add_tensor(other)
        else:
            raise NotImplementedError()

    def __add__(self, other):
        return self.add(other)
    
    def add_sparse_matrix_(self, other):
        assert (self.shape() == other.shape()), \
            "Mismatching shapes, self: {} and other: {}".format(self.shape(), other.shape())
        assert self.n_channels == other.n_channels, \
            "Mismatching number of channels, self: {} and other: {}".format(self.n_channels, other.n_channels)
        self.indices = torch.cat((self.indices, other.indices), dim=1)
        self.values = torch.cat((self.values, other.values), dim=0)
        return self.coalesce_()

    def add_tensor_(self, other):
        self.values = self.values + other
        return self
        
    def add_(self, other):
        if type(other) is SparseMatrix:
            return self.add_sparse_matrix_(other)
        elif type(other) == torch.nn.parameter.Parameter or type(other) == torch.Tensor:
            return self.add_tensor_(other)

    def __iadd__(self, other):
        return self.add_(other)

    def __matmul__(self, other):
        values_out = self.values @ other
        new_shape = (self.n, self.m, values_out.shape[1])
        return SparseMatrix(self.indices, values_out, new_shape, self.indices_diag, self.is_set)

    def __rmatmul__(self, other):
        values_out = other @ self.values
        new_shape = (self.n, self.m, values_out.shape[1])
        return SparseMatrix(self.indices, values_out, new_shape, self.indices_diag, self.is_set)

    def identity(self, mask=None):
        out = self.clone()
        if mask != None:
            out.values = (mask.float() * out.values.T).T
        return out

    def gather_transpose(self, mask):
        indices_out, values_out = torch_sparse.transpose(self.indices, self.values,
                                                         self.n, self.m, coalesced=True)
        return values_out[mask]
        
    def gather_mask(self, mask):
        return self.values[mask]

    def broadcast_from_mask(self, data, mask, device=None):
        '''
        Broadcast data into self shape along indices from index_str
        '''
        idx_overlap = mask.nonzero().T
        n_channels= data.shape[-1]
        
        out_values = torch.zeros(self.nnz(), n_channels).to(device)        
        out_values.scatter_(0, idx_overlap.expand(n_channels, -1).T, data)
        out_shape = (self.n, self.m, n_channels)
        return SparseMatrix(self.indices.clone(), out_values, out_shape, self.indices_diag, self.is_set)

    def transpose(self, transpose_mask=None):
        indices_out, values_out = torch_sparse.transpose(self.indices, self.values,
                                                         self.n, self.m, coalesced=True)
        
        if transpose_mask is not None:
            values_out = (transpose_mask.float() * values_out.T).T
        shape_out = (self.m, self.n, self.n_channels)

        return SparseMatrix(indices_out, values_out, shape_out, self.indices_diag, self.is_set)

    
    def coalesce_(self, op='add'):
        '''
        Add all duplicated entries together
        '''
        self.indices, self.values = torch_sparse.coalesce(self.indices, self.values,
                                                          self.n, self.m, op)
        self.indices_diag = self.calc_indices_diag()
        return self
        
    def coalesce(self, op='add'):
        '''
        Add all duplicated entries together
        '''
        indices, values = torch_sparse.coalesce(self.indices, self.values,
                                                          self.n, self.m, op)
        return SparseMatrix(indices, values, self.shape(), is_set=self.is_set)


    def pool(self, index_str, device=None, op='add'):
        '''
        Pool self.values along indices specified by index_str
        '''
        assert index_str in {"row","col","diag", "all"}
        assert op in {"add", "max", "mean"}
        values = self.values
        if index_str == "all" or index_str == "diag":
            pooled_output = torch.zeros(1, self.n_channels).to(device)
            if index_str == "diag":
                assert self.n == self.m, "Diag only implemented for square matrices"
                values = self.gather_diag(device)
            if op == "add" or op == "mean":
                pooled_output = torch.sum(values, dim=0)
            elif op == "max":
                pooled_output = torch.max(values, dim=0)[0]
            if op == "mean":
                if index_str == "all":
                    denom = self.nnz()
                elif index_str == "diag":
                    denom = self.indices_diag.shape[0]
                if denom == 0:
                    denom = 1.
                pooled_output = pooled_output / denom

        else:
            if index_str == "row":
                indices = self.indices[1]
                n_segments = self.m
            elif index_str == "col":
                indices = self.indices[0]
                n_segments = self.n
            if op == "add":
                pooled_output = torch_scatter.scatter_add(values.T, indices, dim_size=n_segments).T
            elif op == "mean":
                pooled_output = torch_scatter.scatter_mean(values.T, indices, dim_size=n_segments).T
            elif op == "max":
                pooled_output = torch_scatter.scatter_max(values.T, indices, dim_size=n_segments)[0].T

        return pooled_output

    def gather_diag(self, device=None):
        '''
        Gather diagonal elements into dense array of values
        '''
        assert self.n == self.m
        n_diag = self.indices_diag.shape[0]
        out_vals = self.values[self.indices_diag, :]
        out_idx = self.indices[0, self.indices_diag].view(n_diag, 1).expand(n_diag, self.n_channels)
        gathered_output = torch.zeros(self.n, self.n_channels).to(device)
        return gathered_output.scatter_(0, out_idx, out_vals)

    def broadcast(self, data, index_str, device=None):
        '''
        Broadcast data into self shape along indices from index_str
        '''
        assert index_str in {"diag", "row","col", "all"}
        n_channels= data.shape[-1]
        if index_str == "diag":
            data = data.expand(self.n, n_channels)
            return self.embed_diag(data, device)
        elif index_str == "row":
            indices = self.indices[1]
        elif index_str == "col":
            indices = self.indices[0]
        elif index_str == "all":
            indices = torch.zeros(self.nnz(), dtype=torch.int64).to(device)
            data = data.unsqueeze(0)
        idx_0 = indices.shape[0]
        idx_tensor_expanded = indices.view(idx_0, 1).expand(idx_0, n_channels)
        out_values = torch.gather(data, 0, idx_tensor_expanded)
        
        out_shape = (self.n, self.m, out_values.shape[1])
        return SparseMatrix(self.indices.clone(), out_values, out_shape, self.indices_diag, self.is_set)
    
    def embed_diag(self, data, device=None):
        '''
        Assuming self is square matrix, embed data onto diagonal where indices
        exist  in self for diagonal elements
        '''
        assert self.n == self.m == data.shape[0], "Requires square matrix and dense data for whole diagonal"
        n_channels = data.shape[1]
        diag_indices = self.indices.T[self.indices_diag][:,0]
        data_out = data[diag_indices]
        out_values = torch.zeros(self.nnz(), n_channels).to(device)
        
        out_values.scatter_(0, self.indices_diag.expand(n_channels, -1).T, data_out)
        out_shape = (self.n, self.m, n_channels)
        return SparseMatrix(self.indices.clone(), out_values, out_shape, self.indices_diag, self.is_set)


    def index_sort(self):
        '''
        Sort index by lexicographical order
        '''
        idx = self.indices.T
        M = idx.max()
        augmented_idx = idx.select(1, 0) * M + idx.select(1, 1)
        ind = augmented_idx.sort().indices
        
        indices = idx.index_select(0, ind).T
        #values = self.values.index_select(0, ind)
        return indices
    
    def is_sorted(self):
        return (self.index_sort() == self.indices).all()

    def clone(self):
        if self.indices_diag != None:
            indices_diag = self.indices_diag.clone()
        else:
            indices_diag = None
        return SparseMatrix(self.indices.clone(), self.values.clone(),
                            self.shape(), indices_diag, self.is_set)

    def zero_(self):
        self.values = self.values.zero_()
        return self

    def to_torch_sparse(self):
        return torch.sparse_coo_tensor(self.indices, self.values,
                                   size=self.shape())

    def to_dense(self):
        out = self.to_torch_sparse().to_dense()
        permute_channel_dim = [-1] + list(range(self.ndimension()))
        return out.permute(*permute_channel_dim)
