# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from src.DataSchema import DataSchema
from src.EquivariantLayer import EquivariantLayer
from src.SparseMatrixEquivariantLayer import SparseMatrixEquivariantLayer, \
                    SparseMatrixEntityPoolingLayer, SparseMatrixEntityBroadcastingLayer
from src.SparseEquivariantLayer import SparseEquivariantLayer
from src.Modules import RelationNorm, Activation, EntityPooling, \
                        EntityBroadcasting, Dropout
import pdb

class EquivariantNetwork(nn.Module):
    '''
    Equivariant Network for dense tensor inputs and outputs
    schema: source schema of input data
    input_channels: number of channels in input tensor for each relation
    source_layers: channels of layers fully equivariant to source schema
    target_layers: channels of layers fully equivariant to target schema
    output_dim: number of dimensions of output tensor
    schema_out: schema representing target tensor
    activation: layer activations for source and target
    dropout: rate of dropout at each layer
    pool_op: 'mean' or 'add' pooling at each layer
    norm: whether to normalize output of each layer
    norm_affine: if normalizing output, whether norm is affine
    final_activation: activation of output of last layer
    '''
    def __init__(self, schema, input_channels, source_layers=[32,64,32],
                 target_layers=[32], output_dim=1, schema_out=None,
                 activation=F.relu, dropout=0, pool_op='mean',
                 norm=True, norm_affine=True, final_activation=nn.Identity()):
        super(EquivariantNetwork, self).__init__()
        self.schema = schema
        if schema_out == None:
            self.schema_out = schema
        else:
            self.schema_out = schema_out
        self.input_channels = input_channels

        self.activation = activation
        self.source_activation = Activation(schema, self.activation)
        self.target_activation = Activation(self.schema_out, self.activation)

        self.dropout = Dropout(p=dropout)
        self.source_dropout  = Activation(self.schema, self.dropout)
        self.target_dropout = Activation(self.schema_out, self.dropout)
        # Equivariant layers with source schema
        self.n_source_layers = len(source_layers)
        self.source_layers = nn.ModuleList([])
        self.source_layers.append(EquivariantLayer(
                self.schema, input_channels, source_layers[0], pool_op=pool_op))
        self.source_layers.extend([
                EquivariantLayer(self.schema, source_layers[i-1], source_layers[i], pool_op=pool_op)
                for i in range(1, len(source_layers))])
        if norm:
            self.source_norms = nn.ModuleList()
            for channels in source_layers:
                norm_dict = nn.ModuleDict()
                for relation in self.schema.relations:
                    norm_dict[str(relation.id)] = nn.GroupNorm(channels, channels, affine=norm_affine)
                norm_activation = Activation(self.schema, norm_dict, is_dict=True)
                self.source_norms.append(norm_activation)
        else:
            self.source_norms = nn.ModuleList([Activation(schema, nn.Identity())
                                        for _ in source_layers])

        # Equivariant layers with target schema
        target_layers = target_layers + [output_dim]
        self.n_target_layers = len(target_layers)
        self.target_layers = nn.ModuleList([])
        self.target_layers.append(EquivariantLayer(self.schema, source_layers[-1],
                                                   target_layers[0],
                                                   schema_out=self.schema_out,
                                                   pool_op=pool_op))
        self.target_layers.extend([
                EquivariantLayer(self.schema_out, target_layers[i-1],
                                 target_layers[i], pool_op=pool_op)
                for i in range(1, len(target_layers))])
        if norm:
            self.target_norms = nn.ModuleList()
            for channels in target_layers:
                norm_dict = nn.ModuleDict()
                for relation in self.schema_out.relations:
                    norm_dict[str(relation.id)] = nn.GroupNorm(channels, channels, affine=norm_affine)
                norm_activation = Activation(self.schema_out, norm_dict, is_dict=True)
                self.target_norms.append(norm_activation)
        else:
            self.target_norms = nn.ModuleList([Activation(self.schema_out, nn.Identity())
                                        for _ in target_layers])

        self.final_activation = final_activation
        self.final_rel_activation = Activation(self.schema_out, self.final_activation)

    def forward(self, data):
        for i in range(self.n_source_layers):
            data = self.source_dropout(self.source_activation(self.source_norms[i](
                    self.source_layers[i](data))))
        for i in range(self.n_target_layers - 1):
            data = self.target_dropout(self.target_activation(self.target_norms[i](
                    self.target_layers[i](data))))
        data = self.target_layers[-1](data)
        out = self.final_rel_activation(data)
        return out


class EquivariantAutoEncoder(nn.Module):
    def __init__(self, schema, encoding_dim = 10):
        super(EquivariantAutoEncoder, self).__init__()
        self.schema = schema
        self.encoding_dim = encoding_dim
        self.dropout_rate = 0.2
        self.hidden_dims = [64]*2
        self.all_dims = [1] + list(self.hidden_dims) + [self.encoding_dim]
        self.n_layers = len(self.all_dims)

        self.ReLU = Activation(schema, nn.ReLU())
        self.Dropout = Activation(schema, nn.Dropout(p=self.dropout_rate))

        encoder = []
        encoder.append(EquivariantLayer(self.schema, self.all_dims[0], self.all_dims[1]))
        for i in range(1, self.n_layers-1):
            encoder.append(self.ReLU)
            encoder.append(RelationNorm(self.schema, self.all_dims[i], affine=False))
            encoder.append(self.Dropout)
            encoder.append(EquivariantLayer(self.schema, self.all_dims[i], self.all_dims[i+1]))
        encoder.append(EntityPooling(schema, self.encoding_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        decoder.append(EntityBroadcasting(schema, self.encoding_dim))
        decoder.append(EquivariantLayer(self.schema, self.all_dims[-1], self.all_dims[-2]))
        for i in range(self.n_layers-2, 0, -1):
            decoder.append(self.ReLU)
            decoder.append(RelationNorm(self.schema, self.all_dims[i], affine=False))
            decoder.append(self.Dropout)
            decoder.append(EquivariantLayer(self.schema, self.all_dims[i], self.all_dims[i-1]))
        self.decoder = nn.Sequential(*decoder)

    def get_encoding_size(self):
        return {entity.id: (entity.n_instances, self.hidden_dim) 
                for entity in self.schema.entities}

    def forward(self, data):
        enc  = self.encoder(data)
        out = self.decoder(enc)
        return out

class SparseEquivariantNetwork(nn.Module):
    def __init__(self, schema, n_channels, target_rel=None, final_activation=None):
        super(SparseEquivariantNetwork, self).__init__()
        self.schema = schema
        self.n_channels = n_channels
        self.hidden_dims = (32, 64, 32)
        self.all_dims = [n_channels] + list(self.hidden_dims) + [n_channels]
        self.target_rel = target_rel
        self.ReLU = Activation(schema, nn.ReLU(), is_sparse=True)
        sequential = []
        for i in range(1, len(self.all_dims)-1):
            sequential.append(SparseEquivariantLayer(self.schema, self.all_dims[i-1], self.all_dims[i]))
            sequential.append(self.ReLU)
            sequential.append(RelationNorm(self.schema, self.all_dims[i], affine=False, sparse=True))
        sequential.append(SparseEquivariantLayer(self.schema, self.all_dims[-2],
                                                 self.all_dims[-1], target_rel=target_rel))

        self.sequential = nn.Sequential(*sequential)

        if final_activation == None:
            final_activation = nn.Identity()
        self.final = Activation(schema, final_activation, is_sparse=True)

    def forward(self, data):
        out = self.sequential(data)
        if self.classification:
            out = out[self.target_rel].to_dense()[0,:]
        out = self.final(out)
        return out

class SparseMatrixEntityPredictor(nn.Module):
    '''
    Network for predicting properties of a single entity, where relations
    take the form of sparse matrices
    '''
    def __init__(self, schema, input_channels, activation=F.relu, 
                 layers=[32, 64, 32], target_entities=None,
                 fc_layers=[], final_activation=nn.Identity(),
                 output_dim=1,  dropout=0, norm=True, pool_op='mean', norm_affine=False):
        super(SparseMatrixEntityPredictor, self).__init__()
        self.schema = schema
        self.input_channels = input_channels

        self.activation = activation
        self.rel_activation = Activation(schema, self.activation, is_sparse=True)

        self.dropout = Dropout(p=dropout)
        self.rel_dropout  = Activation(schema, self.dropout, is_sparse=True)

        # Equivariant Layers
        self.n_equiv_layers = len(layers)
        self.equiv_layers = nn.ModuleList([])
        self.equiv_layers.append(SparseMatrixEquivariantLayer(
                schema, input_channels, layers[0], pool_op=pool_op))
        self.equiv_layers.extend([
                SparseMatrixEquivariantLayer(schema, layers[i-1], layers[i], pool_op=pool_op)
                for i in range(1, len(layers))])
        if norm:
            self.norms = nn.ModuleList()
            for channels in layers:
                norm_dict = nn.ModuleDict()
                for relation in self.schema.relations:
                    norm_dict[str(relation.id)] = nn.BatchNorm1d(channels, affine=norm_affine, track_running_stats=False)
                norm_activation = Activation(schema, norm_dict, is_dict=True, is_sparse=True)
                self.norms.append(norm_activation)
        else:
            self.norms = nn.ModuleList([Activation(schema, nn.Identity(), is_sparse=True)
                                        for _ in layers])

        # Entity embeddings
        embedding_layers = fc_layers + [output_dim]
        self.pooling = SparseMatrixEntityPoolingLayer(schema, layers[-1],
                                                      embedding_layers[0],
                                                      entities=target_entities,
                                                      pool_op=pool_op)
        self.n_fc_layers = len(fc_layers)
        self.fc_layers = nn.ModuleList([])
        self.fc_layers.extend([nn.Linear(embedding_layers[i-1], embedding_layers[i])
                            for i in range(1, self.n_fc_layers+1)])

        self.final_activation = final_activation


    def forward(self, data, idx_identity=None, idx_transpose=None, data_out=None, get_embeddings=False):
        if idx_identity is None or idx_transpose is None:
            print("Calculating idx_identity and idx_transpose. This can be precomputed.")
            idx_identity, idx_transpose = data.calculate_indices()
        for i in range(self.n_equiv_layers):
            data = self.rel_dropout(self.rel_activation(self.norms[i](
                    self.equiv_layers[i](data, idx_identity, idx_transpose))))
        data = self.pooling(data, data_out)
        out = data[0].values
        if self.n_fc_layers > 0 and get_embeddings == False:
            out = self.fc_layers[0](out)
            for i in range(1, self.n_fc_layers):
                out = self.fc_layers[i](self.dropout(self.activation(out)))
        out = self.final_activation(out)
        return out

class SparseMatrixAutoEncoder(nn.Module):
    '''
    Autoencoder to produce entity embeddings which can be used for
    node classification / regression or link prediction
    '''
    def __init__(self, schema, input_channels=1, activation=F.relu,
                 layers=[64, 64, 64], embedding_dim=50,
                 dropout=0, norm=True, pool_op='mean', norm_affine=False,
                 final_activation=nn.Identity(),
                 embedding_entities = None,
                 output_relations = None):
        super(SparseMatrixAutoEncoder, self).__init__()
        self.schema = schema
        if output_relations == None:
            self.schema_out = schema
        else:
            self.schema_out = DataSchema(schema.entities, output_relations)
        self.input_channels = input_channels

        self.activation = activation
        self.rel_activation = Activation(schema, self.activation, is_sparse=True)

        self.dropout = Dropout(p=dropout)
        self.rel_dropout  = Activation(schema, self.dropout, is_sparse=True)

        self.n_equiv_layers = len(layers)
        self.equiv_layers = nn.ModuleList([])
        self.equiv_layers.append(SparseMatrixEquivariantLayer(
                schema, input_channels, layers[0], pool_op=pool_op))
        self.equiv_layers.extend([
                SparseMatrixEquivariantLayer(schema, layers[i-1], layers[i], pool_op=pool_op)
                for i in range(1, len(layers))])
        if norm:
            self.norms = nn.ModuleList()
            for channels in layers:
                norm_dict = nn.ModuleDict()
                for relation in self.schema.relations:
                    norm_dict[str(relation.id)] = nn.BatchNorm1d(channels, affine=norm_affine, track_running_stats=False)
                norm_activation = Activation(schema, norm_dict, is_dict=True, is_sparse=True)
                self.norms.append(norm_activation)
        else:
            self.norms = nn.ModuleList([Activation(schema, nn.Identity(), is_sparse=True)
                                        for _ in layers])

        # Entity embeddings
        self.pooling = SparseMatrixEntityPoolingLayer(schema, layers[-1],
                                                      embedding_dim,
                                                      entities=embedding_entities,
                                                      pool_op=pool_op)
        self.broadcasting = SparseMatrixEntityBroadcastingLayer(self.schema_out,
                                                                embedding_dim,
                                                                input_channels,
                                                                entities=embedding_entities,
                                                                pool_op=pool_op)

        self.final_activation = Activation(schema, final_activation, is_sparse=True)



    def forward(self, data, idx_identity=None, idx_transpose=None,
                data_target=None, data_embedding=None, get_embeddings=False):
        if idx_identity is None or idx_transpose is None:
            print("Calculating idx_identity and idx_transpose. This can be precomputed.")
            idx_identity, idx_transpose = data.calculate_indices()
        for i in range(self.n_equiv_layers):
            data = self.rel_dropout(self.rel_activation(self.norms[i](
                    self.equiv_layers[i](data, idx_identity, idx_transpose))))
        data = self.pooling(data, data_embedding)
        if get_embeddings:
            return data
        data = self.broadcasting(data, data_target)
        data = self.final_activation(data)
        return data