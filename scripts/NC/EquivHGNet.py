# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from src.DataSchema import DataSchema
from src.EquivariantLayer import EquivariantLayer
from src.SparseMatrixEquivariantLayer import SparseMatrixEquivariantLayer, \
                    SparseMatrixEntityPoolingLayer, SparseMatrixEntityBroadcastingLayer
from src.SparseEquivariantLayer import SparseEquivariantLayer
from src.Modules import Activation,  Dropout,  SparseMatrixRelationLinear


class EquivHGNet(nn.Module):
    '''
    Network for predicting properties of a single entity, where relations
    take the form of sparse matrices
    '''
    def __init__(self, schema, input_channels, activation=F.relu, 
                 layers=[32, 64, 32], target_entities=None,
                 fc_layers=[], final_activation=nn.Identity(),
                 output_dim=1,  dropout=0, norm=True, pool_op='mean',
                 in_fc_layer=True, norm_affine=False):
        super(EquivHGNet, self).__init__()

        self.schema = schema
        self.input_channels = input_channels

        self.activation = activation
        self.rel_activation = Activation(schema, self.activation, is_sparse=True)

        self.dropout = Dropout(p=dropout)
        self.rel_dropout  = Activation(schema, self.dropout, is_sparse=True)

        self.use_in_fc_layer = in_fc_layer
        # Equivariant Layers
        self.equiv_layers = nn.ModuleList([])
        if self.use_in_fc_layer:
            # Simple fully connected layers for input attributes
            self.fc_in_layer = SparseMatrixRelationLinear(schema, self.input_channels,
                                                          layers[0])
            self.n_equiv_layers = len(layers) - 1
        else:
            # Alternatively, use an equivariant layer
            self.equiv_layers.append(SparseMatrixEquivariantLayer(
                schema, input_channels, layers[0], pool_op=pool_op))
            self.n_equiv_layers = len(layers)

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
        if self.use_in_fc_layer:
            data = self.fc_in_layer(data)
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

