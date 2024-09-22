import torch
import utils as u
import os
import numpy as np

class Dataset():
    def __init__(self,args):
        #args.email_eu_args = u.Namespace(args.email_eu_args)
        args.dataset_args = u.Namespace(args.dataset)

        edges_file = os.path.join(args.dataset_args.folder, args.dataset_args.edges_file)  
        #print(edges_file)

        self.edges = self.load_edges(args, edges_file)
    
    def load_edges(self, args, edges_file):
        data = u.load_data(edges_file, starting_line=1, sep=',')
        cols = u.Namespace({'idx1': 0,
                            'source': 1,
                            'target': 2,
                            'time': 3,
                            'label': 4,
                            'idx2': 5
                            })
        data = data.long()

        #add the reversed link to make the graph undirected
        data = torch.cat([data,data[:,[cols.idx1,
                                          cols.target,
                                          cols.source,
                                          cols.time,
                                          cols.label,
                                          cols.idx2]]],
                        dim=0)

        index = torch.eq(data[:,cols.source],data[:,cols.target])
        # if torch.sum(index) > 0:
        #     print('exists diagonal edges!')

        _, data[:,[cols.source,cols.target]] = data[:,[cols.source,cols.target]].unique(return_inverse = True)
        self.num_nodes = int(data[:,[cols.source,cols.target]].max()+1)

        ids = data[:,cols.source] * self.num_nodes + data[:,cols.target]
        num_unique_edges = len(np.unique(ids))/2

        #first id should be 0 (they are already contiguous)
        data[:,[cols.source,cols.target]] -= 1

        data[:,cols.time] = u.aggregate_by_time(data[:,cols.time],
                                	args.dataset_args.aggr_time)

        _, data[:,[cols.source,cols.target]] = data[:,[cols.source,cols.target]].unique(return_inverse = True)
        self.num_nodes = int(data[:,[cols.source,cols.target]].max()+1)

        ids = data[:,cols.source] * self.num_nodes + data[:,cols.target]

        idx = data[:,[cols.source,
                   	  cols.target,
                   	  cols.time]]
        
        self.max_time = data[:,cols.time].max()
        self.min_time = data[:,cols.time].min()

        return {'idx': idx, 'vals': torch.ones(idx.size(0))}




