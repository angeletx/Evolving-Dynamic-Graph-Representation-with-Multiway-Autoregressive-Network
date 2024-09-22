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
                #print('logname: '.format(self.log_name))

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
        self.mse_loss = torch.nn.MSELoss(reduction="mean")

    def get_log_file_name(self):
        return self.log_name

    def log_epoch_start(self, epoch, num_minibatches, set, minibatch_log_interval=None):
        #ALDO
        self.epoch = epoch
        ######
        self.set = set
        self.losses = []
        self.MRRs = []
        self.GMAUCs = []
        self.MAPs = []
        self.NDCGs = []
        self.new_edges_praucs = []
        self.new_edges_ratios = []
        self.previous_edges_aucs = []
        self.previous_edges_praucs = []
        self.previous_edges_ratios = []

        self.batch_sizes=[]
        self.minibatch_done = 0
        self.num_minibatches = num_minibatches
        if minibatch_log_interval is not None:
            self.minibatch_log_interval = minibatch_log_interval
        logging.info('################ '+set+' epoch '+str(epoch)+' ###################')
        self.lasttime = time.monotonic()
        self.ep_time = self.lasttime

    def get_MSE(self,predictions,ground_truth):
        self.mse_loss(prediction, ground_truth)

    def get_NDCG(self,predictions,true_classes, do_softmax=False):
        if do_softmax:
            probs = torch.softmax(predictions,dim=1)[:,1]
        else:
            probs = predictions

        predictions_np = probs.detach().cpu().numpy()
        true_classes_np = true_classes.detach().cpu().numpy()

        return ndcg_score(true_classes_np.reshape(1,-1), predictions_np.reshape(1,-1))

    def get_GMAUC(self, predictions, true_classes, prev_edges_mask, do_softmax=False):
        if do_softmax:
            probs = torch.softmax(predictions,dim=1)[:,1]
        else:
            probs = predictions

        probs = probs.detach().cpu().numpy()
        true_classes = true_classes.detach().cpu().numpy()
        prev_edges_mask = prev_edges_mask.detach().cpu().numpy()

        previous_edges_probs = probs[prev_edges_mask]
        previous_edges_truth = true_classes[prev_edges_mask]
        new_edges_probs = probs[~prev_edges_mask]
        new_edges_truth = true_classes[~prev_edges_mask]

        new_edges_P = np.sum(new_edges_truth)
        new_edges_ratio = np.float64(new_edges_P/len(new_edges_truth))
        #print("new_edges_ratio: ", new_edges_ratio)

        if new_edges_P == 0 or len(new_edges_truth) == 0:
            new_edges_prauc = 0
            partA = 0
        else:
            new_edges_prauc = average_precision_score(new_edges_truth, new_edges_probs)
            partA = abs(new_edges_prauc - new_edges_ratio)/(1-new_edges_ratio)

        if(len(np.unique(previous_edges_truth)) == 1):
            print('unique previous_edges_truth: ', np.unique(previous_edges_truth))
            previous_edges_auc = 0.5 
        else:
            previous_edges_auc = roc_auc_score(previous_edges_truth, previous_edges_probs)
        
        previous_edges_P = np.sum(previous_edges_truth)
        previous_edges_ratio = np.float64(previous_edges_P/len(previous_edges_truth))
        #print("previous_edges_ratio: ", previous_edges_ratio)

        if previous_edges_P == 0 or len(previous_edges_truth) == 0:
            previous_edges_prauc = 0
        else:
            previous_edges_prauc = average_precision_score(previous_edges_truth, previous_edges_probs)

        product = np.float64(partA * 2* abs(previous_edges_auc-0.5))
        gmauc = product ** 0.5
        
        #if self.set in ['TEST']:
        #    new_positive, new_negative, old_positive, old_negative, new_positive_ratio, old_positive_ratio = tu.get_statistics(true_classes, prev_edges_mask)
        #    print('new_positive: {}, new_negative: {}, old_positive: {}, old_negative: {}, new_positive_ratio: {}, old_positive_ratio: {}'.format(new_positive, new_negative, old_positive, old_negative, new_positive_ratio, old_positive_ratio))

        return gmauc,new_edges_prauc,new_edges_ratio,previous_edges_auc,previous_edges_prauc,previous_edges_ratio
        #return gmauc
        
    def get_MAP(self,predictions,true_classes, do_softmax=False):
        if do_softmax:
            probs = torch.softmax(predictions,dim=1)[:,1]
        else:
            probs = predictions

        predictions_np = probs.detach().cpu().numpy()
        true_classes_np = true_classes.detach().cpu().numpy()

        return average_precision_score(true_classes_np, predictions_np)

    def getMicroAUC(self, probs, true_classes):
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(true_classes.ravel(), probs.ravel())
        return auc(fpr["micro"], tpr["micro"])

    def getMulticlassAUC(self, probs, y):
        n_classes = y.shape[1]
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        return roc_auc["micro"], roc_auc["macro"]
    
#    def getMacroAUC(self, probs, true_classes, avg='macro', multi='ovo'):
#        predictions_np = probs.detach().cpu().numpy()
#        true_classes_np = true_classes.detach().cpu().numpy()
#        
#        return roc_auc_score(true_classes_np, predictions_np, average=avg, multi_class=multi)
    
    def getBinaryAUC(self, probs, true_classes):
        predictions_np = probs.detach().cpu().numpy()
        true_classes_np = true_classes.detach().cpu().numpy()
        
        fpr, tpr, thresholds = roc_curve(true_classes_np, predictions_np)
        #return roc_auc_score(true_classes_np, predictions_np, average=avg, multi_class=multi)
        return auc(fpr, tpr)

    def get_dprime(self, auc):
        standard_normal = stats.norm()
        d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
        return d_prime
        
    #def log_minibatch(self, predictions, true_classes, loss, **kwargs):
    def log_minibatch(self, predictions, label_sp, loss, **kwargs):
        #print(self.args.task)

        if self.args.task == "link_pred":
            # Add to remove diagonal edges
            nondiag_mask = label_sp['idx'][0] != label_sp['idx'][1]
            #print('nondiag_mask.device:',nondiag_mask.device) 
            #label_sp['idx'] = label_sp['idx'][:, nondiag_mask]
            #label_sp['vals'] = label_sp['vals'][nondiag_mask]

            true_ids = label_sp['idx'][:, nondiag_mask]
            true_classes = label_sp['vals'][nondiag_mask]
            predictions = predictions[nondiag_mask]
            
            ##!!! Pay attention!!! This is only for binary classification, even for edge classification. You need to rewrite if applied to multi-class edge classification!!
            probs = torch.softmax(predictions,dim=1)[:,1]
            
            if self.set in ['TEST', 'VALID'] and self.args.task == 'link_pred':
                MRR = self.get_MRR(probs,true_classes, true_ids, do_softmax=False) 
                #mask = torch.BoolTensor(kwargs['prev_edges_mask'])[nondiag_mask]
                mask = torch.BoolTensor(kwargs['prev_edges_mask'])[nondiag_mask.to('cpu')]
                #GMAUC = torch.tensor(self.get_GMAUC(probs,true_classes, mask[nondiag_mask], do_softmax=False))
            
                #if self.set in ['TEST']:
                #    new_positive, new_negative, old_positive, old_negative, new_positive_ratio, old_positive_ratio = tu.get_statistics(true_classes, mask)
                #    print('new_positive: {}, new_negative: {}, old_positive: {}, old_negative: {}, new_positive_ratio: {}, old_positive_ratio: {}'.format(new_positive, new_negative, old_positive, old_negative, new_positive_ratio, old_positive_ratio))

                GMAUC,new_edges_prauc,new_edges_ratio,previous_edges_auc,previous_edges_prauc,previous_edges_ratio = torch.tensor(self.get_GMAUC(probs,true_classes, mask, do_softmax=False))
            else:
                MRR = torch.tensor([0.0])
                GMAUC = torch.tensor([0.0])
                new_edges_prauc = torch.tensor([0.0])
                new_edges_ratio = torch.tensor([0.0])
                previous_edges_auc = torch.tensor([0.0])
                previous_edges_prauc = torch.tensor([0.0])
                previous_edges_ratio = torch.tensor([0.0])

            MAP = torch.tensor(self.get_MAP(probs,true_classes, do_softmax=False))
            NDCG = torch.tensor(self.get_NDCG(probs,true_classes, do_softmax=False))
        
            if len(true_classes.unique()) == 0:
                microAUC = torch.tensor(0)
                macroAUC = torch.tensor(0)
                microDprime = torch.tensor(0)
                macroDprime = torch.tensor(0)
            elif len(true_classes.unique()) == 2:
                AUC = torch.tensor(self.getBinaryAUC(probs, true_classes))
                dprime = torch.tensor(self.get_dprime(AUC))
                microAUC = AUC
                macroAUC = AUC
                microDprime = dprime
                macroDprime = dprime
            else:
                y = label_binarize(true_classes, classes=unique(true_classes))
                microAUC_value, macroAUC_value = self.getMulticlassAUC(probs, y)
                microAUC = torch.tensor(microAUC_value)
                macroAUC = torch.tensor(macroAUC_value) 
                microDprime = torch.tensor(self.get_dprime(microAUC))
                macroDprime = torch.tensor(self.get_dprime(macroAUC))
            
            batch_size = predictions.size(0)
            self.batch_sizes.append(batch_size)

        self.losses.append(loss) 
        self.MRRs.append(MRR)
        self.MAPs.append(MAP)
        self.NDCGs.append(NDCG)
        self.new_edges_praucs.append(new_edges_prauc)
        self.new_edges_ratios.append(new_edges_ratio)
        self.previous_edges_aucs.append(previous_edges_auc)
        self.previous_edges_praucs.append(previous_edges_prauc)
        self.previous_edges_ratios.append(previous_edges_ratio)
                
        self.minibatch_done+=1
        if self.minibatch_done%self.minibatch_log_interval==0 and self.num_classes > 1:
            mb_MRR = self.calc_epoch_metric(self.batch_sizes, self.MRRs)
            mb_MAP = self.calc_epoch_metric(self.batch_sizes, self.MAPs)
            mb_NDCG = self.calc_epoch_metric(self.batch_sizes, self.NDCGs)
            mb_new_edges_prauc = self.calc_epoch_metric(self.batch_sizes, self.new_edges_praucs)
            mb_new_edges_ratio = self.calc_epoch_metric(self.batch_sizes, self.new_edges_ratios)
            mb_previous_edges_auc = self.calc_epoch_metric(self.batch_sizes, self.previous_edges_aucs)
            mb_previous_edges_prauc = self.calc_epoch_metric(self.batch_sizes, self.previous_edges_praucs)
            mb_previous_edges_ratio = self.calc_epoch_metric(self.batch_sizes, self.previous_edges_ratios)
            
            partial_losses = torch.stack(self.losses)
            logging.info(self.set+ ' batch %d / %d - partial NDCG %0.4f' % (self.minibatch_done, self.num_minibatches, mb_NDCG))
            logging.info(self.set+ ' batch %d / %d - partial new_edges_prauc %0.4f - partial new_edges_ratio %0.4f' % (self.minibatch_done, self.num_minibatches, mb_new_edges_prauc, mb_new_edges_ratio))
            logging.info(self.set+ ' batch %d / %d - partial previous_edges_auc %0.4f - partial previous_edges_prauc %0.4f - partial previous_edges_ratio %0.4f' % (self.minibatch_done, self.num_minibatches, mb_previous_edges_auc, mb_previous_edges_prauc, mb_previous_edges_ratio))
            logging.info (self.set+' batch %d / %d - Batch time %d ' % (self.minibatch_done, self.num_minibatches, (time.monotonic()-self.lasttime) ))

        self.lasttime=time.monotonic()

        return probs, true_classes, true_ids #label_sp['vals']

    def floatTensor2String(self, tensorList):
        result = ''
        array_numpy = np.array(tensorList)
        for element in array_numpy:
            result = result + str(element) + ','
        return result

    def log_epoch_done(self,atte = None):
        #print('target_measure: {}'.format(self.args.target_measure))
        eval_measure = 0

        self.losses = torch.stack(self.losses).detach().cpu().numpy()
        #print("self.losses: ", self.losses)
        loss = self.losses.mean().item()
        logging.info(self.set+' mean losses '+ str(loss))
        if self.args.target_measure=='loss' or self.args.target_measure=='Loss':
            eval_measure = -loss
            #print("eval_measure: ", eval_measure)

        if self.args.task == 'link_pred':
            epoch_MRR = self.calc_epoch_metric(self.batch_sizes, self.MRRs)
            epoch_MAP = self.calc_epoch_metric(self.batch_sizes, self.MAPs)
            epoch_NDCG = self.calc_epoch_metric(self.batch_sizes, self.NDCGs)
            epoch_new_edges_prauc = self.calc_epoch_metric(self.batch_sizes, self.new_edges_praucs)
            epoch_new_edges_ratio = self.calc_epoch_metric(self.batch_sizes, self.new_edges_ratios)
            epoch_previous_edges_auc = self.calc_epoch_metric(self.batch_sizes, self.previous_edges_aucs)
            epoch_previous_edges_prauc = self.calc_epoch_metric(self.batch_sizes, self.previous_edges_praucs)
            epoch_previous_edges_ratio = self.calc_epoch_metric(self.batch_sizes, self.previous_edges_ratios)
        
            #print('epoch_MAP: {}'.format(epoch_MAP))
            logging.info(self.set+' mean MRR '+ str(epoch_MRR)+' - mean MAP '+ str(epoch_MAP))
            logging.info(self.set+' mean NDCG '+ str(epoch_NDCG))
            logging.info(self.set+' mean new_edges_prauc '+ str(epoch_new_edges_prauc)+' - mean new_edges_ratio '+ str(epoch_new_edges_ratio))
            logging.info(self.set+' mean previous_edges_auc '+ str(epoch_previous_edges_auc)+' - mean previous_edges_prauc '+ str(epoch_previous_edges_prauc)+' - mean previous_edges_ratio '+ str(epoch_previous_edges_ratio))

            if self.set in ['TEST']:
                logging.info('Snapshot Loss ' + self.floatTensor2String(self.losses))
                logging.info('Snapshot MAP ' + self.floatTensor2String(self.MAPs))
                logging.info('Snapshot MRR ' + self.floatTensor2String(self.MRRs))
                logging.info('Snapshot NDCG ' + self.floatTensor2String(self.NDCGs))
                logging.info('Snapshot new_edges_prauc ' + self.floatTensor2String(self.new_edges_praucs))
                logging.info('Snapshot previous_edges_auc ' + self.floatTensor2String(self.previous_edges_aucs))
                logging.info('Snapshot previous_edges_prauc ' + self.floatTensor2String(self.previous_edges_praucs))
                logging.info('Snapshot previous_edges_ratio ' + self.floatTensor2String(self.previous_edges_ratios))
                logging.info('Snapshot new_edges_ratio ' + self.floatTensor2String(self.previous_edges_ratios))

            if self.args.target_measure=='MRR' or self.args.target_measure=='mrr':
                eval_measure = epoch_MRR
            if self.args.target_measure=='MAP' or self.args.target_measure=='map':
                eval_measure = epoch_MAP
            if self.args.target_measure=='NDCG' or self.args.target_measure=='ndcg':
                eval_measure = epoch_NDCG
        
        logging.info (self.set+' Total epoch time: '+ str(((time.monotonic()-self.ep_time))))
        #print('final eval_measure: {}'.format(eval_measure))

        return eval_measure

    def get_MRR(self,predictions,true_classes, adj ,do_softmax=False):
        if do_softmax:
            probs = torch.softmax(predictions,dim=1)[:,1]
        else:
            probs = predictions

        probs = probs.cpu().numpy()
        true_classes = true_classes.cpu().numpy()
        adj = adj.cpu().numpy()

        #print('probs length: {}, adj[0]: {}, adj[1]: {}'.format(np.shape(probs),adj[0][-1],adj[1][-1]))
        pred_matrix = coo_matrix((probs,(adj[0],adj[1]))).toarray()
        true_matrix = coo_matrix((true_classes,(adj[0],adj[1]))).toarray()

        row_MRRs = []
        for i,pred_row in enumerate(pred_matrix):
            #check if there are any existing edges
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
                                   dtype=float)[ordered_existing_mask]

        MRR = (1/existing_ranks).sum()/existing_ranks.shape[0]
        return MRR

    def calc_epoch_metric(self,batch_sizes, metric_val):
        batch_sizes = torch.tensor(batch_sizes, dtype = torch.float)
        epoch_metric_val = torch.stack(metric_val).cpu() * batch_sizes
        epoch_metric_val = epoch_metric_val.sum()/batch_sizes.sum()

        return epoch_metric_val.detach().item()
