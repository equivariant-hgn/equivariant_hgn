import sys
sys.path.append('../../')
import time
import argparse

from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score, auc, roc_auc_score, precision_recall_curve
from data_lp import load_data, get_train_valid_pos, get_train_neg, \
    get_valid_neg, get_valid_neg_2hop, get_test_neigh_from_file, gen_file_for_evaluate
from EquivHGAE import EquivHGAE, EquivLinkPredictor
from src.SparseMatrix import SparseMatrix
from src.DataSchema import DataSchema, SparseMatrixData, Relation
import warnings
import pdb
warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")

#%%
def set_seed(seed):
    random.seed(seed, version=2)
    np.random.seed(random.randint(0, 2**32))
    torch.manual_seed(random.randint(0, 2**32))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_features(data, schema, feats_type):
    '''
    TODO: IMPLEMENT THIS
    '''
    # Select features for nodes
    in_dims = {}
    num_relations = len(schema.relations) - len(schema.entities)

    if feats_type == 0:
        # Keep all node attributes
        pass
    '''
    elif feats_type == 1:
        # Set all non-target node attributes to zero
        for ent_i in schema.entities:
            if ent_i.id != target_ent:
                # 10 dimensions for some reason
                n_dim = 10
                rel_id = num_relations + ent_i.id
                data[rel_id] = SparseMatrix.from_other_sparse_matrix(data[rel_id], n_dim)

    # Don't even worry bout it
    features_list = [mat2tensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2[]
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    '''
    for rel in schema.relations:
        in_dims[rel.id] = data[rel.id].n_channels
    return data, in_dims

def loss_fcn(data_pred, data_true):
    return F.binary_cross_entropy(data_pred, data_true)
    
def make_target_matrix(relation, pos_head, pos_tail, neg_head, neg_tail, device):
    n_pos = pos_head.shape[0]
    pos_indices = np.vstack((pos_head, pos_tail))
    pos_values = np.ones((n_pos, 1))
    n_neg = neg_head.shape[0]
    neg_indices = np.vstack((neg_head, neg_tail))
    neg_values = np.zeros((n_neg, 1))
    indices = torch.LongTensor(np.concatenate((pos_indices, neg_indices), 1))
    values = torch.FloatTensor(np.concatenate((pos_values, neg_values), 0))
    shape = (relation.entities[0].n_instances,
             relation.entities[1].n_instances, 1)
    data_target = SparseMatrix(indices=indices, values=values, shape=shape)
    data_target = data_target.to(device).coalesce_()

    return data_target

def combine_matrices(matrix_a, matrix_b):
    '''
    Given Matrix A (target) and Matrix B (supplement), would like to get a
    matrix that includes the indices of both matrix A and B, but only the
    values of Matrix A. Additionally, return a mask indicating which indices
    corresponding to Matrix A.
    '''
    # We only want indices from matrix_b, not values
    matrix_b_zero = matrix_b.clone()
    matrix_b_zero.values.zero_()
    matrix_combined = matrix_a + matrix_b_zero

    # To determine indices corresponding to matrix_a, make binary matrix
    matrix_a_ones = matrix_a.clone()
    matrix_a_ones.values.zero_()
    matrix_a_ones.values += 1
    mask_matrix_combined = matrix_a_ones + matrix_b_zero
    matrix_a_mask = mask_matrix_combined.values[:,0].nonzero().squeeze()
    return matrix_combined, matrix_a_mask

def coalesce_matrix(matrix):
    coalesced_matrix = matrix.coalesce(op='mean')
    left = coalesced_matrix.indices[0,:]
    right = coalesced_matrix.indices[1,:]
    labels = coalesced_matrix.values.squeeze()
    return coalesced_matrix, left, right, labels

def make_combined_data(schema, input_data, target_rel_id, target_matrix):
    '''
    given dataset and a single target matrix for predictions, produce new dataset
    with indices combining original dataset with new target matrix's indices
    '''
    combined_data = input_data.clone()
    combined_data[target_rel_id] += target_matrix
    return combined_data


def make_target_matrix_test(relation, left, right, labels, device):
    indices = torch.LongTensor(np.vstack((left, right)))
    values = torch.FloatTensor(labels).unsqueeze(1)
    shape = (relation.entities[0].n_instances,
             relation.entities[1].n_instances, 1)
    return SparseMatrix(indices=indices, values=values, shape=shape).to(device)




#%%
def run_model(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_equiv = args.decoder == 'equiv'

    # Collect data and schema
    schema, schema_out, data_original, dl = load_data(args.dataset, use_edge_data=args.use_edge_data)
    data, in_dims = select_features(data_original, schema, args.feats_type)
    data = data.to(device)
    
    # Precompute data indices
    indices_identity, indices_transpose = data.calculate_indices()
    # Get target relations and create data structure for embeddings
    target_rel_ids = dl.links_test['data'].keys()
    target_rels = [schema.relations[rel_id] for rel_id in target_rel_ids]
    target_ents = schema.entities
    if use_equiv:
        output_rel = None
    else:
        # Currently only equivariant decoder works for amazon dataset
        if len(target_rels) > 1:
            raise NotImplementedError()
        else:
            output_rel = target_rels[0]
    data_embedding = SparseMatrixData.make_entity_embeddings(target_ents,
                                                             args.embedding_dim)
    data_embedding.to(device)

    # Get training and validation positive samples now
    train_pos_heads, train_pos_tails = dict(), dict()
    val_pos_heads, val_pos_tails = dict(), dict()
    for target_rel_id in target_rel_ids:
        train_val_pos = get_train_valid_pos(dl, target_rel_id)
        train_pos_heads[target_rel_id], train_pos_tails[target_rel_id], \
            val_pos_heads[target_rel_id], val_pos_tails[target_rel_id] = train_val_pos

    # Create network and optimizer
    net = EquivLinkPredictor(schema, in_dims,
                    layers=args.layers,
                    embedding_dim=args.embedding_dim,
                    embedding_entities=target_ents,
                    output_rel=output_rel,
                    activation=eval('nn.%s()' % args.act_fn),
                    final_activation = nn.Identity(),
                    dropout=args.dropout,
                    pool_op=args.pool_op,
                    norm_affine=args.norm_affine,
                    in_fc_layer=args.in_fc_layer,
                    decode = args.decoder)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Set up logging and checkpointing
    if args.wandb_log_run:
        wandb.init(config=args,
            settings=wandb.Settings(start_method='fork'),
            project='XXX',
            entity='XXX')
        wandb.watch(net, log='all', log_freq=args.wandb_log_param_freq)
    print(args)
    run_name = args.dataset + '_' + str(args.run)
    if args.wandb_log_run and wandb.run.name is not None:
        run_name = run_name + '_' + str(wandb.run.name)
    if args.checkpoint_path != '':
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = f"checkpoint/checkpoint_{run_name}.pt"
    print("Checkpoint Path: " + checkpoint_path)
    val_metric_best = -1e10

    # training
    loss_func = nn.BCELoss()
    progress = tqdm(range(args.epoch), desc="Epoch 0", position=0, leave=True)
    for epoch in progress:
        net.train()
        # Make target matrix and labels to train on
        if use_equiv:
            # Target is same as input
            target_schema = schema
            data_target = data.clone()
        else:
            # Target is just target relation
            target_schema = DataSchema(schema.entities, target_rels)
            data_target = SparseMatrixData(target_schema)
        labels_train = torch.Tensor([]).to(device)
        for target_rel in target_rels:
            train_neg_head, train_neg_tail = get_train_neg(dl, target_rel.id)
            train_matrix = make_target_matrix(target_rel,
                                              train_pos_heads[target_rel.id],
                                              train_pos_tails[target_rel.id],
                                              train_neg_head, train_neg_tail,
                                              device)
            data_target[target_rel.id] = train_matrix
            labels_train_rel = train_matrix.values.squeeze()
            labels_train = torch.cat([labels_train, labels_train_rel])

        # Make prediction
        if use_equiv:
            idx_id_tgt, idx_trans_tgt = data_target.calculate_indices()
            output_data = net(data, indices_identity, indices_transpose,
                       data_embedding, data_target, idx_id_tgt, idx_trans_tgt)
            logits_combined = torch.Tensor([]).to(device)
            for target_rel in target_rels:
                logits_rel = output_data[target_rel.id].values.squeeze()
                logits_combined = torch.cat([logits_combined, logits_rel])
        else:
            logits_combined = net(data, indices_identity, indices_transpose,
                         data_embedding, data_target)
        logp = torch.sigmoid(logits_combined)
        train_loss = loss_func(logp, labels_train)

        # autograd
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Update logging
        progress.set_description(f"Epoch {epoch}")
        progress.set_postfix(loss=train_loss.item())
        wandb_log = {'Train Loss': train_loss.item(), 'epoch':epoch}

        # Evaluate on validation set
        net.eval()
        if epoch % args.val_every == 0:
            with torch.no_grad():
                net.eval()
                left = torch.Tensor([]).to(device)
                right = torch.Tensor([]).to(device)
                labels_val = torch.Tensor([]).to(device)
                valid_masks = {}
                for target_rel in target_rels:
                    if args.val_neg == '2hop':
                        valid_neg_head, valid_neg_tail = get_valid_neg_2hop(dl, target_rel.id)
                    else:
                        valid_neg_head, valid_neg_tail = get_valid_neg(dl, target_rel.id)
                    valid_matrix_full = make_target_matrix(target_rel,
                                                     val_pos_heads[target_rel.id], val_pos_tails[target_rel.id],
                                                     valid_neg_head, valid_neg_tail,
                                                     device)
                    valid_matrix, left_rel, right_rel, labels_val_rel = coalesce_matrix(valid_matrix_full)
                    left = torch.cat([left, left_rel])
                    right = torch.cat([right, right_rel])
                    labels_val = torch.cat([labels_val, labels_val_rel])
                    if use_equiv:
                        # Add in training indices
                        valid_combined_matrix, valid_mask = combine_matrices(valid_matrix, train_matrix)
                        valid_masks[target_rel.id] = valid_mask
                        data_target[target_rel.id] = valid_combined_matrix
                left = left.cpu().numpy()
                right = right.cpu().numpy()
                edge_list = np.concatenate([left.reshape((1,-1)), right.reshape((1,-1))], axis=0)

                if use_equiv:
                    data_target.zero_()
                    idx_id_val, idx_trans_val = data_target.calculate_indices()
                    output_data = net(data, indices_identity, indices_transpose,
                               data_embedding, data_target, idx_id_val, idx_trans_val)
                    logits_combined = torch.Tensor([]).to(device)
                    for target_rel in target_rels:
                        logits_rel_full = output_data[target_rel.id].values.squeeze()
                        logits_rel = logits_rel_full[valid_masks[target_rel.id]]
                        logits_combined = torch.cat([logits_combined, logits_rel])
                else:
                    # Mult-target not figured out for non-equivariant decoders yet
                    raise NotImplementedError()
                    logits_combined = net(data, indices_identity, indices_transpose,
                                 data_embedding, data_target)
                logp = torch.sigmoid(logits_combined)
                val_loss = loss_func(logp, labels_val).item()

                wandb_log.update({'val_loss': val_loss})
                res = dl.evaluate(edge_list, logp.cpu().numpy(), labels_val.cpu().numpy())
                val_roc_auc = res['roc_auc']
                val_mrr = res['MRR']
                wandb_log.update(res)
                print("\nVal Loss: {:.3f} Val ROC AUC: {:.3f} Val MRR: {:.3f}".format(
                    val_loss, val_roc_auc, val_mrr))
                if args.val_metric == 'loss':
                    val_metric = -val_loss
                elif args.val_metric == 'roc_auc':
                    val_metric = val_roc_auc
                elif args.val_metric == 'mrr':
                    val_metric = val_mrr

                if val_metric > val_metric_best:
                    val_metric_best = val_metric
                    print("New best, saving")
                    torch.save({
                        'epoch': epoch,
                        'net_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss.item(),
                        'val_loss': val_loss,
                        'val_roc_auc': val_roc_auc,
                        'val_mrr': val_mrr
                        }, checkpoint_path)
                    if args.wandb_log_run:
                        wandb.summary["val_roc_auc_best"] = val_roc_auc
                        wandb.summary["val_mrr_best"] = val_mrr
                        wandb.summary["val_loss_best"] = val_loss
                        wandb.summary["epoch_best"] = epoch
                        wandb.summary["train_loss_best"] = train_loss.item()
                        wandb.save(checkpoint_path)
        if args.wandb_log_run:
            wandb.log(wandb_log)

    # Evaluate on test set
    if args.evaluate:
        for target_rel in target_rels:
            print("Evaluating Target Rel " + str(target_rel.id))
            checkpoint = torch.load(checkpoint_path)
            net.load_state_dict(checkpoint['net_state_dict'])
            net.eval()
            if use_equiv:
                # Target is same as input
                data_target = data.clone()
            else:
                # Target is just target relation
                data_target = SparseMatrixData(target_schema)
            with torch.no_grad():
                left_full, right_full, test_labels_full = get_test_neigh_from_file(dl, args.dataset, target_rel.id)
                test_matrix_full =  make_target_matrix_test(target_rel, left_full, right_full,
                                                      test_labels_full, device)
                test_matrix, left, right, test_labels = coalesce_matrix(test_matrix_full)

                if use_equiv:
                    test_combined_matrix, test_mask = combine_matrices(test_matrix, train_matrix)
                    data_target[target_rel.id] = test_combined_matrix
                    data_target.zero_()
                    idx_id_tst, idx_trans_tst = data_target.calculate_indices()
                    data_out = net(data, indices_identity, indices_transpose,
                               data_embedding, data_target, idx_id_tst, idx_trans_tst)
                    logits_full = data_out[target_rel.id].values.squeeze()
                    logits = logits_full[test_mask]
                else:
                    data_target[target_rel.id] = test_matrix
                    logits = net(data, indices_identity, indices_transpose,
                                 data_embedding, data_target)
                pred = torch.sigmoid(logits).cpu().numpy()
                left = left.cpu().numpy()
                right = right.cpu().numpy()
                edge_list = np.vstack((left,right))
                edge_list_full = np.vstack((left_full, right_full))
                file_path = f"test_out/{run_name}.txt"
                gen_file_for_evaluate(dl, edge_list_full, edge_list, pred, target_rel.id,
                                         file_path=file_path)

#%%
def get_hyperparams(argv):
    ap = argparse.ArgumentParser(allow_abbrev=False, description='EquivHGN for Node Classification')
    ap.add_argument('--feats_type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--batch_size', type=int, default=100000)
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training \
                    and testing for N times. Default is 1.')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--dataset', type=str, default='PubMed')
    ap.add_argument('--checkpoint_path', type=str, default='')
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--embedding_dim', type=int, default=64)
    ap.add_argument('--fc_layer', type=int, default=64)
    #ap.add_argument('--fc_layers', type=str, nargs='*', default=[64],
    #                    help='Fully connected layers for target embeddings')
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--act_fn', type=str, default='LeakyReLU')
    ap.add_argument('--in_fc_layer', type=int, default=1)
    ap.add_argument('--optimizer', type=str, default='Adam')
    ap.add_argument('--val_every', type=int, default=5)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--norm_affine', type=int, default=1)
    ap.add_argument('--pool_op', type=str, default='mean')
    ap.add_argument('--use_edge_data',  type=int, default=0)
    ap.add_argument('--save_embeddings', dest='save_embeddings', action='store_true', default=True)
    ap.add_argument('--no_save_embeddings', dest='save_embeddings', action='store_false', default=True)
    ap.set_defaults(save_embeddings=True)
    ap.add_argument('--wandb_log_param_freq', type=int, default=100)
    ap.add_argument('--wandb_log_loss_freq', type=int, default=1)
    ap.add_argument('--wandb_log_run', dest='wandb_log_run', action='store_true',
                        help='Log this run in wandb')
    ap.add_argument('--wandb_no_log_run', dest='wandb_log_run', action='store_false',
                        help='Do not log this run in wandb')
    ap.add_argument('--output', type=str)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--evaluate', type=int, default=1)
    ap.add_argument('--decoder', type=str, default='equiv')
    ap.add_argument('--val_neg', type=str, default='random')
    ap.add_argument('--val_metric', type=str, default='roc_auc')
    ap.set_defaults(wandb_log_run=False)
    args, argv = ap.parse_known_args(argv)
    if args.output == None:
        args.output = args.dataset + '_emb.dat'
    if args.in_fc_layer == 1:
        args.in_fc_layer = True
    else:
        args.in_fc_layer = False
    if args.evaluate == 1:
        args.evaluate = True
    else:
        args.evaluate = False
    if args.norm_affine == 1:
        args.norm_affine = True
    else:
        args.norm_affine = False
    if args.use_edge_data == 1:
        args.use_edge_data = True
    else:
        args.use_edge_data = False
    args.layers = [args.width]*args.depth
    if args.fc_layer == 0:
        args.fc_layers = []
    else:
        args.fc_layers = [args.fc_layer]
    return args

    
if __name__ == '__main__':
    argv = sys.argv[1:]
    args = get_hyperparams(argv)
    set_seed(args.seed)
    #%%
    run_model(args)
