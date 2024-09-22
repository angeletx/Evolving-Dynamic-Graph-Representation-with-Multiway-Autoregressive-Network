import utils as u
import os
import numpy as np
import torch

class Dataset():
	def __init__(self,args):

		args.dataset_args = u.Namespace(args.dataset)

		edges_file = os.path.join(args.dataset_args.folder, args.dataset_args.edges_file)  
		self.edges = self.load_edges(args, edges_file)


	def load_edges(self, args, edges_file):
		data = u.load_data(edges_file, starting_line=0, sep=' ')
		cols = u.Namespace({'source': 0,
							 'target': 1,
							 'time': 2})
		data = data.long()

		#add edges in the other direction (symmetric)
		data = torch.cat([data,
						   data[:,[cols.target,
						   		   cols.source,
						   		   cols.time]]],
						   dim=0)

		index = torch.eq(data[:,cols.source],data[:,cols.target])
		#if torch.sum(index) > 0:
		#	print('exists diagonal edges!')

		_, data[:,[cols.source,cols.target]] = data[:,[cols.source,cols.target]].unique(return_inverse = True)
		self.num_nodes = int(data[:,[cols.source,cols.target]].max()+1)

		ids = data[:,cols.source] * self.num_nodes + data[:,cols.target]

		data[:,cols.time] = u.aggregate_by_time(data[:,cols.time],
									args.dataset_args.aggr_time)

		#use only first X time steps
		indices = data[:,cols.time] < args.dataset_args.steps_accounted
		data = data[indices,:]

		_, data[:,[cols.source,cols.target]] = data[:,[cols.source,cols.target]].unique(return_inverse = True)
		self.num_nodes = int(data[:,[cols.source,cols.target]].max()+1)

		ids = data[:,cols.source] * self.num_nodes + data[:,cols.target]
		
		idx = data[:,[cols.source,
				   	  cols.target,
				   	  cols.time]]

		self.max_time = data[:,cols.time].max()
		self.min_time = data[:,cols.time].min()

		return {'idx': idx, 'vals': torch.ones(idx.size(0))}
