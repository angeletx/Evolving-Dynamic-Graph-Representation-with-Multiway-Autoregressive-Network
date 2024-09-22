from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import utils as u

class splitter():
    '''
    creates 3 splits
    train
    dev
    test
    '''
    def __init__(self,args,tasker):
        
        
        if tasker.is_static: #### For static datsets
            assert args.train_proportion + args.dev_proportion < 1, \
                'there\'s no space for test samples'
            
            random_perm=False
            indexes = tasker.data.nodes_with_label
            
            if random_perm:
                perm_idx = torch.randperm(indexes.size(0))
                perm_idx = indexes[perm_idx]
            else:
                print ('tasker.data.nodes',indexes.size())
                perm_idx, _ = indexes.sort()
            
            self.train_idx = perm_idx[:int(args.train_proportion*perm_idx.size(0))]
            self.dev_idx = perm_idx[int(args.train_proportion*perm_idx.size(0)): int((args.train_proportion+args.dev_proportion)*perm_idx.size(0))]
            self.test_idx = perm_idx[int((args.train_proportion+args.dev_proportion)*perm_idx.size(0)):]
            
            train = static_data_split(tasker, self.train_idx, test = False)
            train = DataLoader(train, shuffle=True,**args.data_loading_params)
            
            dev = static_data_split(tasker, self.dev_idx, test = True)
            dev = DataLoader(dev, shuffle=False,**args.data_loading_params)
            
            test = static_data_split(tasker, self.test_idx, test = True)
            test = DataLoader(test, shuffle=False,**args.data_loading_params)
                        
            self.tasker = tasker
            self.train = train
            self.dev = dev
            self.test = test
            
            
        else: #### For datsets with time
            assert args.train_proportion + args.dev_proportion < 1, \
                'there\'s no space for test samples'
            start = tasker.data.min_time + args.num_hist_steps 
            end = args.train_proportion
            
            end = int(np.floor((tasker.data.max_time.type(torch.float)+1) * end))
            train = data_split(tasker, start, end, test = False)
            train = DataLoader(train,**args.data_loading_params)       
    
            start = end
            end = args.dev_proportion + args.train_proportion
            end = int((np.floor(tasker.data.max_time.type(torch.float)+1) * end))
            if args.task == 'link_pred':
                dev = data_split(tasker, start, end, test = True, all_edges=True)
                #dev = data_split(tasker, start, end, test = True)
            else:
                dev = data_split(tasker, start, end, test = True)
            dev = DataLoader(dev,num_workers=args.data_loading_params['num_workers'])

            start = end
            
            #the +1 is because I assume that max_time exists in the dataset
            end = int(tasker.max_time) + 1
            if args.task == 'link_pred':
                test = data_split(tasker, start, end, test = True, all_edges=True)
            else:
                test = data_split(tasker, start, end, test = True)
            test = DataLoader(test,num_workers=args.data_loading_params['num_workers'])
                        
            self.tasker = tasker
            self.train = train
            self.dev = dev
            self.test = test

class data_split(Dataset):
    def __init__(self, tasker, start, end, test, **kwargs):
        '''
        start and end are indices indicating what items belong to this split
        '''
        self.tasker = tasker
        self.start = start
        self.end = end
        self.test = test
        self.kwargs = kwargs

    def __len__(self):
        #print('start: {}, end: {}'.format(self.start, self.end))
        return self.end-self.start

    def __getitem__(self,idx):
        idx = self.start + idx
        t = self.tasker.get_sample(idx, test = self.test, **self.kwargs)
        return t


class static_data_split(Dataset):
    def __init__(self, tasker, indexes, test):
        '''
        start and end are indices indicating what items belong to this split
        '''
        self.tasker = tasker
        self.indexes = indexes
        self.test = test
        self.adj_matrix = tasker.adj_matrix

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self,idx):
        #print('idx: {}'.fomat(idx))
        idx = self.indexes[idx]
        return self.tasker.get_sample(idx,test = self.test)
