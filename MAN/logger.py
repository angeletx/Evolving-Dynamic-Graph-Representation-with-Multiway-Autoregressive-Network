import logging
import pprint
import sys
import datetime
import torch
import utils
import matplotlib.pyplot as plt
import time
from sklearn.metrics import average_precision_score, roc_auc_score, auc, roc_curve, ndcg_score
from scipy.sparse import coo_matrix
import numpy as np
import os
from scipy import stats
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import taskers_utils as tu

class Logger():
    def __init__(self, args, num_classes, minibatch_log_interval=10):

        if args is not None:
            if not hasattr(args, 'log'):
                currdate=str(datetime.datetime.today().strftime('%Y%m%d%H%M%S'))
                self.log_name= 'log/log_'+args.dataset['name']+'_'+args.task+'_'+args.model+'_'+currdate+'_r'+str(args.rank)+'.log'
            else:
                log_folder = args.log['folder']
                if not os.path.exists(log_folder):
                    os.makedirs(log_folder)
                self.log_name = os.path.join(log_folder, args.log['file'])

            if args.use_log:
                print ("Log file:", self.log_name)
                logging.basicConfig(filename=self.log_name, level=logging.INFO)
            else:
                print ("Log: STDOUT")
                logging.basicConfig(stream=sys.stdout, level=logging.INFO)

            logging.info ('*** PARAMETERS ***')

            logging.info (pprint.pformat(args.__dict__)) # displays the string
            logging.info ('')
        else:
            print ("Log: STDOUT")
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        self.num_classes = num_classes
        self.minibatch_log_interval = minibatch_log_interval
        self.eval_k_list = [10, 100, 1000]
        self.args = args


    def get_log_file_name(self):
        return self.log_name

    def log_epoch_start(self, epoch, num_minibatches, set, minibatch_log_interval=None):
        #ALDO
        self.epoch = epoch
        ######
        self.set = set
        self.losses = []
        self.MRRs = []
        self.MAPs = []
        self.NDCGs = []

        self.batch_sizes=[]
        self.minibatch_done = 0
        self.num_minibatches = num_minibatches
        if minibatch_log_interval is not None:
            self.minibatch_log_interval = minibatch_log_interval
        logging.info('################ '+set+' epoch '+str(epoch)+' ###################')
        self.lasttime = time.monotonic()
        self.ep_time = self.lasttime

    def get_NDCG(self,predictions,true_classes, do_softmax=False):
        if do_softmax:
            probs = torch.softmax(predictions,dim=1)[:,1]
        else:
            probs = predictions

        predictions_np = probs.detach().cpu().numpy()
        true_classes_np = true_classes.detach().cpu().numpy()

        return ndcg_score(true_classes_np.reshape(1,-1), predictions_np.reshape(1,-1))
        
    def get_MAP(self,predictions,true_classes, do_softmax=False):
        if do_softmax:
            probs = torch.softmax(predictions,dim=1)[:,1]
        else:
            probs = predictions

        predictions_np = probs.detach().cpu().numpy()
        true_classes_np = true_classes.detach().cpu().numpy()

        return average_precision_score(true_classes_np, predictions_np)

    def get_dprime(self, auc):
        standard_normal = stats.norm()
        d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
        return d_prime
        
    def log_minibatch(self, predictions, label_sp, loss, **kwargs):

        # Add to remove diagonal edges
        nondiag_mask = label_sp['idx'][0] != label_sp['idx'][1]
        label_sp['idx'] = label_sp['idx'][:, nondiag_mask]
        label_sp['vals'] = label_sp['vals'][nondiag_mask]

        true_classes = label_sp['vals']
        predictions = predictions[nondiag_mask]

        probs = torch.softmax(predictions,dim=1)[:,1]
        
        if self.set in ['TEST', 'VALID'] and self.args.task == 'link_pred':
            MRR = self.get_MRR(probs,true_classes, label_sp['idx'],do_softmax=False) 
            mask = torch.BoolTensor(kwargs['prev_edges_mask'])[nondiag_mask]    
        else:
            MRR = torch.tensor([0.0])

        MAP = torch.tensor(self.get_MAP(probs,true_classes, do_softmax=False))
        NDCG = torch.tensor(self.get_NDCG(probs,true_classes, do_softmax=False))
        
        batch_size = predictions.size(0)
        self.batch_sizes.append(batch_size)

        self.losses.append(loss) 
        self.MRRs.append(MRR)
        self.MAPs.append(MAP)
        self.NDCGs.append(NDCG)
                
        self.minibatch_done+=1
        if self.minibatch_done%self.minibatch_log_interval==0:
            mb_MRR = self.calc_epoch_metric(self.batch_sizes, self.MRRs)
            mb_MAP = self.calc_epoch_metric(self.batch_sizes, self.MAPs)
            mb_NDCG = self.calc_epoch_metric(self.batch_sizes, self.NDCGs)
                        
            partial_losses = torch.stack(self.losses)
            logging.info(self.set+ ' batch %d / %d - partial MRR  %0.4f - partial MAP %0.4f' % (self.minibatch_done, self.num_minibatches, mb_MRR, mb_MAP))
            logging.info(self.set+ ' batch %d / %d - partial NDCG %0.4f' % (self.minibatch_done, self.num_minibatches, mb_NDCG))
            logging.info (self.set+' batch %d / %d - Batch time %d ' % (self.minibatch_done, self.num_minibatches, (time.monotonic()-self.lasttime) ))

        self.lasttime=time.monotonic()

    def floatTensor2String(self, tensorList):
        result = ''
        array_numpy = np.array(tensorList)
        for element in array_numpy:
            result = result + str(element) + ','
        return result

    def log_epoch_done(self,atte = None):
        eval_measure = 0

        self.losses = torch.stack(self.losses)
        logging.info(self.set+' mean losses '+ str(self.losses.mean()))
        if self.args.target_measure=='loss' or self.args.target_measure=='Loss':
            eval_measure = self.losses.mean()

        epoch_MRR = self.calc_epoch_metric(self.batch_sizes, self.MRRs)
        epoch_MAP = self.calc_epoch_metric(self.batch_sizes, self.MAPs)
        epoch_NDCG = self.calc_epoch_metric(self.batch_sizes, self.NDCGs)

        logging.info(self.set+' mean MRR '+ str(epoch_MRR)+' - mean MAP '+ str(epoch_MAP))
        logging.info(self.set+' mean NDCG '+ str(epoch_NDCG))
        
        if self.args.target_measure=='MRR' or self.args.target_measure=='mrr':
            eval_measure = epoch_MRR
        if self.args.target_measure=='MAP' or self.args.target_measure=='map':
            eval_measure = epoch_MAP
        if self.args.target_measure=='NDCG' or self.args.target_measure=='ndcg':
            eval_measure = epoch_NDCG

        return eval_measure

    def get_MRR(self,predictions,true_classes, adj ,do_softmax=False):
        if do_softmax:
            probs = torch.softmax(predictions,dim=1)[:,1]
        else:
            probs = predictions

        probs = probs.cpu().numpy()
        true_classes = true_classes.cpu().numpy()
        adj = adj.cpu().numpy()

        pred_matrix = coo_matrix((probs,(adj[0],adj[1]))).toarray()
        true_matrix = coo_matrix((true_classes,(adj[0],adj[1]))).toarray()

        row_MRRs = []
        for i,pred_row in enumerate(pred_matrix):
            if np.isin(1,true_matrix[i]):
                row_MRRs.append(self.get_row_MRR(pred_row,true_matrix[i]))

        avg_MRR = torch.tensor(row_MRRs).mean()
        return avg_MRR

    def get_row_MRR(self,probs,true_classes):
        existing_mask = true_classes == 1
        #descending in probability
        ordered_indices = np.flip(probs.argsort())

        ordered_existing_mask = existing_mask[ordered_indices]

        existing_ranks = np.arange(1,
                                   true_classes.shape[0]+1,
                                   dtype=np.float)[ordered_existing_mask]

        MRR = (1/existing_ranks).sum()/existing_ranks.shape[0]
        return MRR

    def calc_epoch_metric(self,batch_sizes, metric_val):
        batch_sizes = torch.tensor(batch_sizes, dtype = torch.float)
        epoch_metric_val = torch.stack(metric_val).cpu() * batch_sizes
        epoch_metric_val = epoch_metric_val.sum()/batch_sizes.sum()

        return epoch_metric_val.detach().item()
