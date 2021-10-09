
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import torch
import numpy as np
from src.SparseTensor import SparseTensor
from src.SparseMatrix import SparseMatrix

class DataSchema:
    def __init__(self, entities, relations):
        self.entities = entities
        self.relations = relations
        
class Entity:
    def __init__(self, entity_id, n_instances):
        self.id = entity_id
        self.n_instances = n_instances

    def __gt__(self, other):
        return self.id > other.id
    
    def __repr__(self):
        return "Entity(id={}, n_instances={})".format(self.id, self.n_instances)
    
    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return self.id == other.id and self.n_instances == other.n_instances

class Relation:
    def __init__(self, relation_id, entities, n_channels=1, is_set=False):
        self.id = relation_id
        self.entities = entities
        self.n_channels = n_channels
        self.is_set = is_set

    def get_shape(self):
        return [entity.n_instances for entity in self.entities]

    def get_n_entries(self):
        return np.prod(self.get_shape())

    def __repr__(self):
        return "Relation(id={}, entities={})".format(self.id, self.entities)

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return self.id == other.id and self.get_shape() == other.get_shape() \
                    and self.entities == other.entities

class Data:
    '''
    Wrapper for a dict mapping relations to tensors of their data
    '''
    def __init__(self, schema, data=None, batch_size=1):
        self.schema = schema
        if data == None:
            self.rel_tensors = {}
        else:
            self.rel_tensors = data
        self.batch_size = batch_size
        self.std_means = None

    def __getitem__(self, relation_id):
        return self.rel_tensors[relation_id]

    def __setitem__(self, relation_id, tensor):
        self.rel_tensors[relation_id] = tensor

    def __delitem__(self, relation_id):
        del self.rel_tensors[relation_id]

    def __iter__(self):
        return iter(self.rel_tensors)

    def __len__(self):
        return len(self.rel_tensors)

    def to(self, *args, **kwargs):
        for k, v in self.rel_tensors.items():
            self.rel_tensors[k] = v.to(*args, **kwargs)
        return self

    def __str__(self):
        return self.rel_tensors.__str__()

    def __repr__(self):
        return self.rel_tensors.__repr__()

    def items(self):
        return self.rel_tensors.items()

    def normalize_data(self):
        '''
        Normalize each relation by the std and mean
        '''
        self.std_means = {}
        for relation in self.schema.relations:
            std, mean = torch.std_mean(self.rel_tensors[relation.id])
            self.rel_tensors[relation.id] = (self.rel_tensors[relation.id] - mean) / std
            self.std_means[relation.id] = (std, mean)
        return self

    def unnormalize_data(self, std_means=None):
        '''
        Apply std and mean to each relation, either from arg or from previous
        application of normalize_data. Updates std_means
        '''
        if std_means == None:
            assert self.std_means != None
            std_means = self.std_means

        for relation in self.schema.relations:
            std, mean = std_means[relation.id]
            self.rel_tensors[relation.id] = std*self.rel_tensors[relation.id] + mean
            std_means[relation.id] = torch.std_mean(self.rel_tensors[relation.id])
        self.std_means = std_means
        return self

    def mask_data(self, observed):
        # Return new Data object with all entries not indexed by observed
        # masked as zeros
        masked_data = {rel_id: observed[rel_id] * data
                       for rel_id, data in self.rel_tensors.items()}
        return Data(self.schema, masked_data, batch_size=self.batch_size)

    def to_sparse(self):
        sparse = {}
        for relation in self.schema.relations:
            dense = self.rel_tensors[relation.id]
            sparse[relation.id] = SparseTensor.from_dense_tensor(dense)
        return  Data(self.schema, sparse, batch_size=self.batch_size)

    def to_sparse_matrix(self):
        sparse = {}
        for relation in self.schema.relations:
            dense = self.rel_tensors[relation.id]
            sparse[relation.id] = SparseMatrix.from_dense_tensor(dense)
        return SparseMatrixData(self.schema, sparse, batch_size=self.batch_size)

    def to_tensor(self):
        # Get maximum multiplicity for each relation
        multiplicities = {}
        for relation in self.schema.relations:
            for entity_i, entity in enumerate(relation.entities):
                # Get multiplicity of that entity
                 multiplicity = relation.entities.count(entity)
                 old_multiplicity = multiplicities.get(entity, 0)
                 multiplicities[entity] = max(old_multiplicity, multiplicity)

        # Build tensor of appropriate shape
        n_channels = len(self.schema.relations)
        tensor = torch.zeros(n_channels)
        i = 1
        # Produce mapping from entity to dimension of tensor
        dim_mapping = {}
        for entity in self.schema.entities:
            dim_mapping[entity] = []
            for _ in range(multiplicities[entity]):
                dim_mapping[entity].append(i)
                new_shape = [-1]*tensor.ndim + [entity.n_instances]
                tensor = tensor.unsqueeze(-1).expand(new_shape).clone()
                i += 1

        # Map dimensions in each relation to dimension in final tensor
        relation_mapping = {}
        for relation in self.schema.relations:
            rel_dim_mappings = {}
            entity_counts = {entity: 0 for entity in relation.entities}
            for entity_i, entity in enumerate(relation.entities):
                rel_dim_mappings[1+entity_i] = dim_mapping[entity][entity_counts[entity]]
                entity_counts[entity] += 1
            relation_mapping[relation.id] = rel_dim_mappings

        for relation in self.schema.relations:
            rel_data = self.rel_tensors[relation.id]

            # Get rid of batching dimension
            assert rel_data.shape[0] == 1
            rel_data = rel_data.squeeze(0)

            permutation = list(range(tensor.ndim))

            mapping = relation_mapping[relation.id]
            for _ in range(tensor.ndim - rel_data.ndim):
                rel_data = rel_data.unsqueeze(-1)

            for k, v in mapping.items():
                permutation[permutation.index(k)] = permutation[v]
                permutation[v] = k

            output_size = tensor[relation.id].shape
            rel_data = rel_data.permute(permutation).squeeze(0)
            tensor[relation.id] = rel_data.expand(output_size)
        return tensor

    def clone(self):
        cloned_data = {}
        for rel_id, data in self.rel_tensors.items():
            cloned_data[rel_id] = data.clone()
        return Data(self.schema, cloned_data, self.batch_size)


class SparseTensorData(Data):
    pass

class SparseMatrixData(Data):

    @classmethod
    def make_entity_embeddings(cls, entities, embedding_dim):
        '''
        Initialize from pytorch's built-in sparse tensor
        '''
        data = {}
        relations = []
        for ent in entities:
            n_ent = ent.n_instances
            data[ent.id] = SparseMatrix(
                    indices = torch.arange(n_ent, dtype=torch.int64).repeat(2,1),
                    values=torch.zeros([n_ent, embedding_dim]),
                    shape=(n_ent, n_ent, embedding_dim),
                    is_set=True)
            relations.append(Relation(ent.id, [ent, ent], is_set=True))

        embedding_schema = DataSchema(entities, relations)

        return cls(embedding_schema, data)

    def clone(self):
        cloned_data = {}
        for rel_id, data in self.rel_tensors.items():
            cloned_data[rel_id] = data.clone()
        return SparseMatrixData(self.schema, cloned_data, self.batch_size)

    def mask_data(self, observed):
        pass

    def to_sparse(self):
        return self

    def to_tensor(self):
        pass


    def calculate_indices(self):
        indices_identity = {}
        indices_transpose = {}
        for r_i, matrix_i in self.rel_tensors.items():
            entity_i_n = self.schema.relations[r_i].entities[0]
            entity_i_m = self.schema.relations[r_i].entities[1]
            for r_j, matrix_j in self.rel_tensors.items():
                idx_id_ij = None
                idx_trans_ij = None
                entity_j_n = self.schema.relations[r_j].entities[0]
                entity_j_m = self.schema.relations[r_j].entities[1]
                if entity_i_n == entity_j_n and entity_i_m == entity_j_m:
                    if r_i == r_j:
                        idx_id_ij = (torch.BoolTensor([True]*matrix_i.nnz()).to(matrix_i.values.device),
                                     torch.BoolTensor([True]*matrix_i.nnz()).to(matrix_i.values.device))
                    else:
                        idx_id_ij = matrix_i.calc_intersection_mask(matrix_j)
                if entity_i_n == entity_j_m and entity_i_m == entity_j_n:
                    if r_i == r_j and self.schema.relations[r_i].is_set:
                        idx_trans_ij = (torch.BoolTensor([True]*matrix_i.nnz()).to(matrix_i.values.device),
                                        torch.BoolTensor([True]*matrix_i.nnz()).to(matrix_i.values.device))
                    else:
                        idx_trans_ij = matrix_i.calc_transpose_intersection_overlap(matrix_j)

                indices_identity[r_i, r_j] = idx_id_ij
                indices_transpose[r_i, r_j] = idx_trans_ij
        return indices_identity, indices_transpose

    def zero_(self):
        for rel in self.schema.relations:
            self.rel_tensors[rel.id].zero_()
        return self