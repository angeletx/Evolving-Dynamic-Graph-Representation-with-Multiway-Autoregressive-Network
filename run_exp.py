import utils as u
import torch
import torch.distributed as dist
import numpy as np
import time
import random
import os
import pprint
import sys

#datasets
import email_eu_dl as ee
import auto_syst_dl as aus
import college_msg_dl as cm
import email_eu_dl as ee
import enron_dl as enron
import trust_dl as trust
import body_dl as body

#taskers
import link_pred_tasker as lpt

#models
import models as mls
import gcn_man

import splitter as sp
import Cross_Entropy_Weighted as cew

import trainer as tr

import logger

def random_param_value(param, param_min, param_max, type='int'):
    if str(param) is None or str(param).lower()=='none':
        if type=='int':
            return random.randrange(param_min, param_max+1)
        elif type=='logscale':
            interval=np.logspace(np.log10(param_min), np.log10(param_max), num=100)
            return np.random.choice(interval,1)[0]
        else:
            return random.uniform(param_min, param_max)
    else:
        return param

def build_random_hyper_params(args):

    args.gcn_parameters['num_seq']=args.num_hist_steps + 1
    
    return args

def build_dataset(args):
    if 'email_eu' in args.dataset['name'] :
        return ee.Dataset(args)
    elif args.dataset['name'] == 'as': #'autonomous_syst':
        return aus.Dataset(args)
    elif args.dataset['name'] == 'college':
        return cm.Dataset(args)
    elif args.dataset['name'] == 'trust':
        return trust.Dataset(args)
    elif args.dataset['name'] == 'enron':
        return enron.Dataset(args)
    elif args.dataset['name'] == 'body':
        return body.Dataset(args)
    else:
        raise NotImplementedError('not implemented')

def build_tasker(args,dataset):
    if args.task == 'link_pred':
        return lpt.Link_Pred_Tasker(args,dataset)
    else:
        raise NotImplementedError('still need to implement the other tasks')

def build_gcn(args,tasker):
    gcn_args = u.Namespace(args.gcn_parameters)
    gcn_args.feats_per_node = tasker.feats_per_node
                    
    if args.model == 'gcn_man':
        return gcn_man.CGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device)
    else:
        raise NotImplementedError('need to finish modifying the models')

def build_classifier(args,tasker):	
    feats_per_node = tasker.feats_per_node
    hidden_feats = args.gcn_parameters['hidden_feats']
    if isinstance(hidden_feats, int):
        hidden_feats = [hidden_feats]
    embed_feats = hidden_feats[-1]

    if 'node_cls' == args.task or 'static_node_cls' == args.task:
        mult = 1
    else:
        mult = 2
    if 'gru' in args.model or 'lstm' in args.model:
        in_feats = args.gcn_parameters['lstm_l2_feats'] * mult
    elif args.model == 'skipfeatsgcn' or args.model == 'skipfeatsegcn_h':
        in_feats = (embed_feats + feats_per_node) * mult
    else:
        in_feats = embed_feats * mult

    return mls.Classifier(args,in_features = in_feats, out_features = tasker.num_classes).to(args.device)

if __name__ == '__main__':
    parser = u.create_parser()
    args = u.parse_args(parser)
    
    log_folder = args.log['folder']
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_name = os.path.join(log_folder, args.log['file'])
    if os.path.exists(log_name):
        print('log {} already exists: skip'.format(log_name))
        sys.exit(0)
            
    global rank, wsize, use_cuda
    args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
    args.device='cpu'
    if args.use_cuda:
        args.device='cuda'
    print ("use CUDA:", args.use_cuda, "- device:", args.device)
    try:
        dist.init_process_group(backend='mpi') #, world_size=4
        rank = dist.get_rank()
        wsize = dist.get_world_size()
        print('Hello from process {} (out of {})'.format(dist.get_rank(), dist.get_world_size()))
        if args.use_cuda:
            torch.cuda.set_device(rank )  # are we sure of the rank+1????
            print('using the device {}'.format(torch.cuda.current_device()))
    except:
        rank = 0
        wsize = 1
        print(('MPI backend not preset. Set process rank to {} (out of {})'.format(rank,
                                                                                   wsize)))
    if args.seed is None and args.seed!='None':
        seed = 123+rank#int(time.time())+rank
    else:
        seed=args.seed#+rank
        
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args.seed=seed
    args.rank=rank
    args.wsize=wsize

    # Assign the requested random hyper parameters
    args = build_random_hyper_params(args)

    #build the dataset
    dataset = build_dataset(args)
    #print('build dataset done!')
    
    #build the tasker
    tasker = build_tasker(args,dataset)
    #build the splitter
    splitter = sp.splitter(args,tasker)
    #build the models		
    gcn = build_gcn(args, tasker)
    #print(args.__dict__)
    classifier = build_classifier(args,tasker)
    cross_entropy = cew.Cross_Entropy(args,dataset).to(args.device)

    #trainer
    trainer = tr.Trainer(args,
                         splitter = splitter,
                         gcn = gcn,
                         classifier = classifier,
                         comp_loss = cross_entropy,
                         dataset = dataset,
                         num_classes = tasker.num_classes)

    trainer.train()
    
