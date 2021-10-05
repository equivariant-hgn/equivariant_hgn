import torch
import torch.nn as nn
import torch.nn.functional as F
from src.DataSchema import DataSchema
from src.EquivariantLayer import EquivariantLayer
from src.SparseMatrixEquivariantLayer import SparseMatrixEquivariantLayer, \
                    SparseMatrixEntityPoolingLayer, SparseMatrixEntityBroadcastingLayer
from src.SparseEquivariantLayer import SparseEquivariantLayer
from src.Modules import Activation,  Dropout,  SparseMatrixRelationLinear
import pdb

class DistMult(nn.Module):
    def __init__(self, num_rel, dim):
        super(DistMult, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(size=(num_rel, dim, dim)))
        nn.init.xavier_normal_(self.W, gain=1.414)

    def forward(self, left_emb, right_emb, r_id):
        thW = self.W[r_id].unsqueeze(0).repeat(len(left_emb), 1, 1)
        left_emb = torch.unsqueeze(left_emb, 1)
        right_emb = torch.unsqueeze(right_emb, 2)
        return torch.bmm(torch.bmm(left_emb, thW), right_emb).squeeze()

class Dot(nn.Module):
    def __init__(self):
        super(Dot, self).__init__()
    def forward(self, left_emb, right_emb, r_id):
        left_emb = torch.unsqueeze(left_emb, 1)
        right_emb = torch.unsqueeze(right_emb, 2)
        return torch.bmm(left_emb, right_emb).squeeze()

class EquivEncoder(nn.Module):
    '''
    Encoder to produce entity embeddings which can be used for
    node classification / regression or link prediction
    '''
    def __init__(self, schema, input_channels=1, activation=F.relu,
                 layers=[64, 64, 64], embedding_dim=50,
                 dropout=0, pool_op='mean', norm_affine=False,
                 embedding_entities = None,
                 in_fc_layer=True):
        super(EquivEncoder, self).__init__()
        self.schema = schema
        self.input_channels = input_channels

        self.activation = activation
        self.rel_activation = Activation(schema, self.activation, is_sparse=True)

        self.dropout = Dropout(p=dropout)
        self.rel_dropout  = Activation(schema, self.dropout, is_sparse=True)

        self.use_in_fc_layer = in_fc_layer
        # Equivariant Layeres
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
        self.norms = nn.ModuleList()
        for channels in layers:
            norm_dict = nn.ModuleDict()
            for relation in self.schema.relations:
                norm_dict[str(relation.id)] = nn.BatchNorm1d(channels, affine=norm_affine, track_running_stats=False)
            norm_activation = Activation(schema, norm_dict, is_dict=True, is_sparse=True)
            self.norms.append(norm_activation)

        # Entity embeddings
        self.pooling = SparseMatrixEntityPoolingLayer(schema, layers[-1],
                                                      embedding_dim,
                                                      entities=embedding_entities,
                                                      pool_op=pool_op)


    def forward(self, data, idx_identity=None, idx_transpose=None,
                data_embedding=None):
        if idx_identity is None or idx_transpose is None:
            print("Calculating idx_identity and idx_transpose. This can be precomputed.")
            idx_identity, idx_transpose = data.calculate_indices()
        if self.use_in_fc_layer:
            data = self.fc_in_layer(data)
        for i in range(self.n_equiv_layers):
            data = self.rel_dropout(self.rel_activation(self.norms[i](
                    self.equiv_layers[i](data, idx_identity, idx_transpose))))
        data = self.pooling(data, data_embedding)
        return data


class EquivDecoder(nn.Module):
    '''
    
    '''
    def __init__(self, schema, activation=F.relu,
                 layers=[64, 64, 64], embedding_dim=50,
                 dropout=0, pool_op='mean', norm_affine=False,
                 embedding_entities = None,
                 output_relations = None,
                 out_fc_layer=True):
        super(EquivDecoder, self).__init__()
        self.schema = schema

        self.activation = activation
        self.rel_activation = Activation(schema, self.activation, is_sparse=True)

        self.dropout = Dropout(p=dropout)
        self.rel_dropout  = Activation(schema, self.dropout, is_sparse=True)

        self.use_out_fc_layer = out_fc_layer

        # Equivariant Layers
        self.broadcasting = SparseMatrixEntityBroadcastingLayer(self.schema,
                                                                embedding_dim,
                                                                layers[0],
                                                                entities=embedding_entities,
                                                                pool_op=pool_op)

        self.equiv_layers = nn.ModuleList([])
        self.equiv_layers.extend([
                SparseMatrixEquivariantLayer(schema, layers[i-1], layers[i], pool_op=pool_op)
                for i in range(1, len(layers))])
        self.n_layers = len(layers) - 1
        if self.use_out_fc_layer:
            # Add fully connected layer to output
            self.fc_out_layer = SparseMatrixRelationLinear(schema, layers[-1], 1)
        else:
            # Alternatively, use an equivariant layer
            self.equiv_layers.append(SparseMatrixEquivariantLayer(
                schema, layers[-1], 1, pool_op=pool_op))

        self.norms = nn.ModuleList()
        for channels in layers:
            norm_dict = nn.ModuleDict()
            for relation in self.schema.relations:
                norm_dict[str(relation.id)] = nn.BatchNorm1d(channels, affine=norm_affine, track_running_stats=False)
            norm_activation = Activation(schema, norm_dict, is_dict=True, is_sparse=True)
            self.norms.append(norm_activation)


    def forward(self, data_embedding, idx_identity=None, idx_transpose=None,
                data_target=None):
        if idx_identity is None or idx_transpose is None:
            print("Calculating idx_identity and idx_transpose. This can be precomputed.")
            idx_identity, idx_transpose = data_target.calculate_indices()

        data = self.broadcasting(data_embedding, data_target)
        for i in range(self.n_layers):
            data = self.rel_dropout(self.rel_activation(self.norms[i](
                    self.equiv_layers[i](data, idx_identity, idx_transpose))))
        if self.use_out_fc_layer:
            data = self.fc_out_layer(data)
        else:
            data = self.equiv_layers[-1](data, idx_identity, idx_transpose)
        return data


class EquivLinkPredictor(nn.Module):
    def __init__(self, schema, input_channels=1, activation=F.relu,
                 layers=[64, 64, 64], embedding_dim=50,
                 dropout=0,  pool_op='mean', norm_affine=False,
                 final_activation=nn.Identity(),
                 embedding_entities = None,
                 output_rel = None,
                 in_fc_layer=True,
                 decode = 'dot'):
        super(EquivLinkPredictor, self).__init__()
        self.output_rel = output_rel
        self.encoder = EquivEncoder(schema, input_channels, activation, layers,
                                    embedding_dim, dropout, pool_op,
                                    norm_affine, embedding_entities, in_fc_layer)
        self.decode = decode
        if self.decode == 'dot':
            self.decoder = Dot()
        elif self.decode == 'distmult':
            self.decoder = DistMult(len(schema.relations), embedding_dim)
        elif decode == 'equiv':
            self.decoder = EquivDecoder(schema, activation,
                 layers, embedding_dim,
                 dropout, pool_op, norm_affine,
                 embedding_entities,
                 output_rel,
                 out_fc_layer=in_fc_layer)
        elif self.decode == 'broadcast':
            if output_rel == None:
                self.schema_out = schema
            else:
                self.schema_out = DataSchema(schema.entities, [output_rel])
            self.decoder = SparseMatrixEntityBroadcastingLayer(self.schema_out,
                                                                embedding_dim,
                                                                input_channels,
                                                                entities=embedding_entities,
                                                                pool_op=pool_op)

        self.final_activation = Activation(schema, final_activation, is_sparse=True)

    def forward(self, data, idx_identity=None, idx_transpose=None,
                data_embedding=None, data_target=None,
                idx_id_out=None, idx_trans_out=None):
        embeddings = self.encoder(data, idx_identity, idx_transpose, data_embedding)
        if self.decode == 'dot' or self.decode == 'distmult':
            left_id = self.output_rel.entities[0].id
            left_target_indices = data_target[self.output_rel.id].indices[0]
            left = embeddings[left_id].values[left_target_indices]
            right_id = self.output_rel.entities[1].id
            right_target_indices = data_target[self.output_rel.id].indices[1]
            right = embeddings[right_id].values[right_target_indices]
            return self.decoder(left, right, self.output_rel.id)
        elif self.decode == 'broadcast':
            return self.decoder(embeddings, data_target)
        elif self.decode == 'equiv':
            return self.decoder(embeddings, idx_id_out, idx_trans_out, data_target)

class EquivHGAE(nn.Module):
    '''
    Autoencoder to produce entity embeddings which can be used for
    node classification / regression or link prediction
    '''
    def __init__(self, schema, input_channels=1, activation=F.relu,
                 layers=[64, 64, 64], embedding_dim=50,
                 dropout=0, norm=True, pool_op='mean', norm_affine=False,
                 final_activation=nn.Identity(),
                 embedding_entities = None,
                 output_relations = None,
                 in_fc_layer=True):
        super(EquivHGAE, self).__init__()
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

        self.use_in_fc_layer = in_fc_layer
        # Equivariant Layeres
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
        if self.use_in_fc_layer:
            data = self.fc_in_layer(data)
        for i in range(self.n_equiv_layers):
            data = self.rel_dropout(self.rel_activation(self.norms[i](
                    self.equiv_layers[i](data, idx_identity, idx_transpose))))
        data = self.pooling(data, data_embedding)
        if get_embeddings:
            return data
        data = self.broadcasting(data, data_target)
        data = self.final_activation(data)
        return data
