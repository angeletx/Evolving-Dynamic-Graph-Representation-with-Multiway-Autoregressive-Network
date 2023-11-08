import numpy as np
import matplotlib.pyplot as plt
import os
from pylab import *
import pprint
import re

##### Parameters ######
filename = sys.argv[-1] # log filename
cl_to_plot_id = 1 # Target class, typically the low frequent one
if 'reddit' in filename or ('bitcoin' in filename and 'edge' in filename):
    cl_to_plot_id = 0 # 0 for reddit dataset or bitcoin edge cls

simulate_early_stop = 0 # Early stop patience
eval_k = 1000 # to compute metrics @K (for instance precision@1000)
print_params = True # Print the parameters of each simulation
##### End parameters ######

if 'elliptic' in filename or 'reddit' in filename or ('bitcoin' in filename and 'edge' in filename):
	target_measure='avg_f1' #'avg_f1' #'f1' # map mrr f1 p r loss avg_p avg_r avg_f1
else:
    target_measure='map' # map mrr f1 p r loss avg_p avg_r avg_f1


# Hyper parameters to analyze
params = []
params.append('learning_rate')
params.append('num_hist_steps')
params.append('l1_weight')
params.append('l2_weight')
params.append('hidden_feats')
params.append('class_weights')
params.append('name')
params.append('model')
params.append('num_hidden_layers')

res_map={}
errors = {}
losses = {}
MRRs = {}
MAPs = {}
NDCGs = {}
best_measure = {}
best_epoch = {}
best_map = {}
best_map_epoch = {}
best_mrr = {}
best_mrr_epoch = {}
best_ndcg = {}
best_ndcg_epoch = {}
    
last_test_ep={}
last_test_ep['MRR'] = '-'
last_test_ep['MAP'] = '-'
last_test_ep['NDCG'] = '-'
last_test_ep['best_epoch'] = -1

sets = ['TRAIN', 'VALID', 'TEST']

summary = {}
summary['logFileName'] = filename
summary['highestMAP'] = -1
summary['highestMRR'] = -1
summary['highestNDCG'] = -1
summary['learningRate'] = -1
summary['historyStep'] = -1
summary['l1Weight'] = -1
summary['model'] = ''
summary['gcnHiddenLayerSize'] = -1
summary['numHiddenLayers'] = -1
summary['dataset'] = ''

for s in sets:
    errors[s] = {}
    losses[s] = {}
    MRRs[s] = {}
    MAPs[s] = {}
    NDCGs[s] = {}

    best_measure[s] = 0
    best_epoch[s] = -1
    best_map[s] = 0
    best_mrr[s] = 0
    best_ndcg[s] = 0
    
str_comments=''
str_comments1=''

exp_params={}
exp_params['l1_weight']   = '[0,0,0,0]'
exp_params['l2_weight']   = '[0,0,0,0]'

print ("Start parsing: ",filename)
with open(filename) as f:
    params_line=True
    readlr=False
    for line in f:
        #print (line)
        line=line.replace('INFO:root:','').replace('\n','')
        if params_line: #print parameters
            if "'learning_rate':" in line:
                   readlr=True
            if not readlr:
                str_comments+=line+'\n'
            else:
                str_comments1+=line+'\n'
            if params_line: #print parameters
                for p in params:
                    str_p='\''+p+'\': '
                    if str_p in line:
                        if '}' in line:
                            line = line.split('}')[0]
                        if 'hidden_feats' in str_p and ('[' in line or ']' in line):
                            exp_params[p]=line.split(str_p)[1].split(']')[0]+']'
                        elif 'l1_weight' in str_p and ('[' in line or ']' in line):
                                exp_params[p]=line.split(str_p)[1].split(']')[0]+']'
                        elif 'l2_weight' in str_p and ('[' in line or ']' in line):
                                exp_params[p]=line.split(str_p)[1].split(']')[0]+']'
                        else:
                            exp_params[p]=line.split(str_p)[1].split(',')[0]
                        #exp_params[p] = eval(exp_params[p])
            if line=='':
                params_line=False

        if 'TRAIN epoch' in line or 'VALID epoch' in line or 'TEST epoch' in line:
            set = line.split(' ')[1]
            epoch = int(line.split(' ')[3])+1
            if set=='TEST':
                last_test_ep['best_epoch'] = epoch
            if epoch==50000:
                break
        elif 'mean errors' in line:
            v=float(line.split('mean errors ')[1])#float(line.split('(')[1].split(')')[0])
            errors[set][epoch]=v
            if target_measure=='errors':
                if v<best_measure[set]:
                #if v>best_measure[set]:
                    best_measure[set]=v
                    best_epoch[set]=epoch
        elif 'mean losses' in line:
            v = float(line.split('(')[1].split(')')[0].split(',')[0])
            losses[set][epoch]=v
            if target_measure=='loss':
                if v>best_measure[set]:
                    best_measure[set]=v
                    best_epoch[set]=epoch
        elif 'mean MRR' in line:
            v = float(line.split('mean MRR ')[1].split(' ')[0])
            MRRs[set][epoch]=v
            if set=='TEST':
                last_test_ep['MRR'] = v
            if v>best_mrr[set]:
                best_mrr[set]=v
            if target_measure=='mrr':
                if v>best_measure[set]:
                    best_measure[set]=v
                    best_epoch[set]=epoch
            if 'mean MAP' in line:
                v=float(line.split('mean MAP ')[1].split(' ')[0])
                MAPs[set][epoch]=v
                if v>best_map[set]:
                    best_map[set]=v
                if target_measure=='map':
                    if v>best_measure[set]:
                        best_measure[set]=v
                        best_epoch[set]=epoch
                if set=='TEST':
                    last_test_ep['MAP'] = v
        elif 'mean NDCG' in line:
            v = float(line.split('mean NDCG ')[1].split(' ')[0])
            NDCGs[set][epoch]=v
            if set=='TEST':
                last_test_ep['NDCG'] = v
            if v>best_ndcg[set]:
                best_ndcg[set]=v
            if target_measure=='ndcg':
                if v>best_measure[set]:
                    best_measure[set]=v
                    best_epoch[set]=epoch
        

if  best_epoch['TEST']<0 and  best_epoch['VALID']<0 or last_test_ep['best_epoch']<1:
    print ('best_epoch<0: -> skip')
    exit(0)

try:
    res_map['model'] = exp_params['model'].replace("'","")
    str_params=(pprint.pformat(exp_params))
    if print_params:
        print ('str_params:\n', str_params)
    if best_epoch['VALID']>=0:
        best_ep = best_epoch['VALID']
        print ('Highest %s values among all epochs: TRAIN  %0.4f \tVALID  %0.4f \tTEST %0.4f' % (target_measure, best_measure['TRAIN'], best_measure['VALID'], best_measure['TEST']))
    else:
        best_ep = best_epoch['TEST']
        print ('Highest %s values among all epochs:\tTRAIN F1 %0.4f\tTEST %0.4f' % (target_measure, best_measure['TRAIN'], best_measure['TEST']))

    use_latest_ep = True
    try:
        res_map['MAP'] = MAPs['TEST'][best_ep]
    except:
        res_map['MAP'] = last_test_ep['MAP']
    try:
        res_map['MRR'] = MRRs['TEST'][best_ep]
    except:
        res_map['MRR'] = last_test_ep['MRR']

    try:
        res_map['NDCG'] = NDCGs['TEST'][best_ep]
    except:
        res_map['NDCG'] = last_test_ep['NDCG']
        
except:
    print('Some error occurred in', filename,' - Epochs read: ',epoch)
    exit(0)

str_results = ''
str_legend = ''
for k, v in res_map.items():
    str_results+=str(v)+','
    str_legend+=str(k)+','
for k, v in exp_params.items():
    str_results+=str(v)+','
    str_legend+=str(k)+','
str_results+=filename.split('/')[1].split('.log')[0]
str_legend+='log_file'

summary['highestMAP'] = best_map['TEST']
summary['highestMRR'] = best_mrr['TEST']
summary['highestNDCG'] = best_ndcg['TEST']
summary['learningRate'] = eval(exp_params['learning_rate'])
summary['historyStep'] = eval(exp_params['num_hist_steps'])
summary['l1Weight'] = exp_params['l1_weight']
summary['l2Weight'] = exp_params['l2_weight']
summary['model'] = eval(exp_params['model'])
summary['dataset'] = eval(exp_params['name'])
print(exp_params['num_hidden_layers'])
summary['numHiddenLayers'] = eval(exp_params['num_hidden_layers'])
summary['gcnHiddenLayerSize'] = eval(exp_params['hidden_feats'])
if isinstance(summary['gcnHiddenLayerSize'], list) and len(summary['gcnHiddenLayerSize']) < 2:
    summary['gcnHiddenLayerSize'] = eval(str(summary['gcnHiddenLayerSize'])+'*'+str(summary['numHiddenLayers']))

pprint.pprint(summary)
