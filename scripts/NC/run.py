import sys
sys.path.append('../../')
sys.path.append('../')

import time
import argparse
import wandb
from tqdm import tqdm
import pdb
from sklearn.metrics import f1_score
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from utils import EarlyStopping
from EquivHGNet import EquivHGNet
from src.SparseMatrix import SparseMatrix

from data_nc import load_data
import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")


def set_seed(seed):
    random.seed(seed, version=2)
    np.random.seed(random.randint(0, 2**32))
    torch.manual_seed(random.randint(0, 2**32))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_features(data, schema, feats_type, target_ent):
    '''
    TODO: IMPLEMENT THIS
    '''
    # Select features for nodes
    in_dims = {}
    num_relations = len(schema.relations) - len(schema.entities)

    if feats_type == 0:
        # Keep all node attributes
        pass
    elif feats_type == 1:
        # Set all non-target node attributes to zero
        for ent_i in schema.entities:
            if ent_i.id != target_ent:
                # 10 dimensions for some reason
                n_dim = 10
                rel_id = num_relations + ent_i.id
                data[rel_id] = SparseMatrix.from_other_sparse_matrix(data[rel_id], n_dim)

    '''
    elif feats_type == 2:
        # Set all non-target node attributes to one-hot vector
        for i in range(0, len(features_list)):
            if i != target_ent:
                dim = features_list[i].shape[0]
                indices = torch.arange(n_instances).unsqueeze(0).repeat(2, 1)
                values = torch.FloatTensor(np.ones(dim))
                features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = np.ones(dim)
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    '''
    for rel in schema.relations:
        in_dims[rel.id] = data[rel.id].n_channels
    return data, in_dims

def regr_fcn(logits, multi_label=False):
    if multi_label:
        return torch.sigmoid(logits)
    else:
        return F.log_softmax(logits, 1)

def loss_fcn(data_pred, data_true, multi_label=False):
    if multi_label:
        return F.binary_cross_entropy(data_pred, data_true)
    else:
        return F.nll_loss(data_pred, data_true)


def f1_scores(logits, target):
    values = logits.argmax(1).detach().cpu()
    micro = f1_score(target.cpu(), values, average='micro')
    macro = f1_score(target.cpu(), values, average='macro')
    return micro, macro

def f1_scores_multi(logits, target):
    values = (logits.detach().cpu().numpy()>0).astype(int)
    micro = f1_score(target, values, average='micro')
    macro = f1_score(target, values, average='macro')
    return micro, macro

def pred_fcn(values, multi_label=False):
    if multi_label:
        pass
    else:        
        values.cpu().numpy().argmax(axis=1)
#%%
def run_model(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    schema, schema_out, data, data_target, labels, train_val_test_idx, dl = load_data(args.dataset, use_edge_data=args.use_edge_data)
    target_entity_id = 0 # True for all current NC datasets
    target_entity = schema.entities[target_entity_id]
    data, in_dims = select_features(data, schema, args.feats_type, target_entity_id)
    if args.multi_label:
        labels = torch.FloatTensor(labels).to(device)
    else:
        labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)
    
    data = data.to(device)
    indices_identity, indices_transpose = data.calculate_indices()
    data_target = data_target.to(device)

    num_classes = dl.labels_train['num_classes']
    net = EquivHGNet(schema, in_dims,
                        layers = args.layers,
                        in_fc_layer=args.in_fc_layer,
                        fc_layers=args.fc_layers,
                        activation=eval('nn.%s()' % args.act_fn),
                        final_activation = nn.Identity(),
                        target_entities=[target_entity],
                        dropout=args.dropout,
                        output_dim=num_classes,
                        norm=args.norm,
                        pool_op=args.pool_op,
                        norm_affine=args.norm_affine)

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)


    if args.wandb_log_run:
        wandb.init(config=args,
            settings=wandb.Settings(start_method='fork'),
            project="XXX",
            entity='XXX')
        wandb.watch(net, log='all', log_freq=args.wandb_log_param_freq)
    print(args)
    run_name = args.dataset + '_' + str(args.run)
    if args.wandb_log_run and wandb.run.name is not None:
        run_name = run_name + '_' + str(wandb.run.name)

    if args.checkpoint_path is not '':
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = f"checkpoint/checkpoint_{run_name}.pt"

    print("Checkpoint Path: " + checkpoint_path)
    progress = tqdm(range(args.epoch), desc="Epoch 0", position=0, leave=True)
    # training loop
    net.train()
    val_micro_best = 0
    for epoch in progress:
        # training
        net.train()
        optimizer.zero_grad()
        logits = net(data, indices_identity, indices_transpose,
                     data_target).squeeze()
        logp = regr_fcn(logits, args.multi_label)
        train_loss = loss_fcn(logp[train_idx], labels[train_idx], args.multi_label)
        train_loss.backward()
        optimizer.step()
        if args.multi_label:
            train_micro, train_macro = f1_scores_multi(logits[train_idx],
                                                 dl.labels_train['data'][train_idx])
        else:
            train_micro, train_macro = f1_scores(logits[train_idx],
                                                 labels[train_idx])
        with torch.no_grad():
            progress.set_description(f"Epoch {epoch}")
            progress.set_postfix(loss=train_loss.item(), micr=train_micro)
            wandb_log = {'Train Loss': train_loss.item(),
                         'Train Micro': train_micro,
                         'Train Macro': train_macro}
            if epoch % args.val_every == 0:
                # validation
                net.eval()
                logits = net(data, indices_identity, indices_transpose, data_target).squeeze()
                logp = regr_fcn(logits, args.multi_label)
                val_loss = loss_fcn(logp[val_idx], labels[val_idx], args.multi_label)
                if args.multi_label:
                    val_micro, val_macro = f1_scores_multi(logits[val_idx],
                                                         dl.labels_train['data'][val_idx])
                else:
                    val_micro, val_macro = f1_scores(logits[val_idx],
                                                         labels[val_idx])
                print("\nVal Loss: {:.3f} Val Micro-F1: {:.3f} \
Val Macro-F1: {:.3f}".format(val_loss, val_micro, val_macro))
                wandb_log.update({'Val Loss': val_loss.item(),
                                  'Val Micro-F1': val_micro, 'Val Macro-F1': val_macro})
                if val_micro > val_micro_best:

                    val_micro_best = val_micro
                    print("New best, saving")
                    torch.save({
                        'epoch': epoch,
                        'net_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss.item(),
                        'train_micro': train_micro,
                        'train_macro': train_macro,
                        'val_loss': val_loss.item(),
                        'val_micro': val_micro,
                        'val_macro': val_macro
                        }, checkpoint_path)
                    if args.wandb_log_run:
                        wandb.summary["val_micro_best"] = val_micro
                        wandb.summary["val_macro_best"] = val_macro
                        wandb.summary["val_loss_best"] = val_loss.item()
                        wandb.summary["epoch_best"] = epoch
                        wandb.summary["train_loss_best"] = train_loss.item()
                        wandb.summary['train_micro_best'] = train_micro,
                        wandb.summary['train_macro_best'] = train_macro,
                        wandb.save(checkpoint_path)

            if epoch % args.wandb_log_loss_freq == 0:
                if args.wandb_log_run:
                    wandb.log(wandb_log, step=epoch)


    # testing with evaluate_results_nc
    if args.evaluate:

        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net_state_dict'])
        net.eval()
        test_logits = []
        with torch.no_grad():
            logits = net(data, indices_identity, indices_transpose,
                         data_target).squeeze()
            test_logits = logits[test_idx]
            if args.multi_label:
                pred = (test_logits.cpu().numpy()>0).astype(int)
            else:
                pred = test_logits.cpu().numpy().argmax(axis=1)
                onehot = np.eye(num_classes, dtype=np.int32)

            file_path = f"test_out/{run_name}.txt"
            dl.gen_file_for_evaluate(test_idx=test_idx, label=pred,
                                     file_path=file_path,
                                     multi_label=args.multi_label)
            if not args.multi_label:
                pred = onehot[pred]
            print(dl.evaluate(pred))
#%%
def get_hyperparams(argv):
    ap = argparse.ArgumentParser(allow_abbrev=False,
                                 description='EquivHGN for Node Classification')
    ap.set_defaults(dataset='PubMed')
    ap.add_argument('--feats_type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' + 
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1,
                    help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--dataset', type=str, default='IMDB')
    ap.add_argument('--checkpoint_path', type=str, default='')
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--layers', type=int, nargs='*', default=['64']*3,
                        help='Number of channels for equivariant layers')
    ap.add_argument('--fc_layer', type=int, default=64)
    #ap.add_argument('--fc_layers', type=str, nargs='*', default=[64],
    #                    help='Fully connected layers for target embeddings')
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--act_fn', type=str, default='LeakyReLU')
    ap.add_argument('--in_fc_layer', type=int, default=1)
    ap.add_argument('--optimizer', type=str, default='Adam')
    ap.add_argument('--val_every', type=int, default=5)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--norm',  type=int, default=1)
    ap.add_argument('--norm_affine', type=int, default=1)
    ap.add_argument('--pool_op', type=str, default='mean')
    ap.add_argument('--use_edge_data',  type=int, default=1)
    ap.add_argument('--save_embeddings', dest='save_embeddings',
                    action='store_true', default=True)
    ap.add_argument('--no_save_embeddings', dest='save_embeddings',
                    action='store_false', default=True)
    ap.set_defaults(save_embeddings=True)
    ap.add_argument('--wandb_log_param_freq', type=int, default=100)
    ap.add_argument('--wandb_log_loss_freq', type=int, default=1)
    ap.add_argument('--wandb_log_run', dest='wandb_log_run', action='store_true',
                        help='Log this run in wandb')
    ap.add_argument('--wandb_no_log_run', dest='wandb_log_run', action='store_false',
                        help='Do not log this run in wandb')
    ap.add_argument('--output', type=str)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--multi_label', default=False, action='store_true',
                    help='multi-label classification. Only valid for IMDb dataset')
    ap.add_argument('--evaluate', type=int, default=1)
    ap.set_defaults(wandb_log_run=False)

    args, argv = ap.parse_known_args(argv)

    if args.output == None:
        args.output = args.dataset + '_emb.dat'
    if args.dataset == 'IMDB':
        args.multi_label = True
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
    if args.norm == 1:
        args.norm = True
    else:
        args.norm = False
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


#%%
if __name__ == '__main__':
    argv = sys.argv[1:]
    args = get_hyperparams(argv)
    set_seed(args.seed)
    run_model(args)
