import utils as u
import os
import numpy as np
import torch

class Dataset():
	def __init__(self,args):

		#args.email_eu_args = u.Namespace(args.email_eu_args)
		args.dataset_args = u.Namespace(args.dataset)

		edges_file = os.path.join(args.dataset_args.folder, args.dataset_args.edges_file)
		features_file = os.path.join(args.dataset_args.folder, args.dataset_args.features_file)    

		self.edges = self.load_edges(args, edges_file)
		self.nodes_feats, self.feats_per_node = self.load_features(args, features_file)

	def load_features(self, args, features_file):
		feats_with_time = u.load_data(features_file, starting_line=0, tensor_const = torch.FloatTensor)
		unique_time = torch.unique(feats_with_time[:,0])
		feats_per_node = int(feats_with_time[0,1].item())
		node_feats = []
		for t in unique_time:
			index = feats_with_time[:,0] == t
			index = index.nonzero()
			node_feats.append(feats_with_time[index, 2:])
		return node_feats, feats_per_node

	def prepare_node_feats(self, node_feats):
		return node_feats.squeeze()

	def load_edges(self, args, edges_file):
		data = u.load_data(edges_file, starting_line=0)
		cols = u.Namespace({'source': 0,
							 'target': 1,
							 'time': 2})
		data = data.long()

		index = torch.eq(data[:,cols.source],data[:,cols.target])
			
		#first id should be 0 (they are already contiguous)
		data[:,[cols.source,cols.target]] -= 1

		_, data[:,[cols.source,cols.target]] = data[:,[cols.source,cols.target]].unique(return_inverse = True)
		self.num_nodes = int(data[:,[cols.source,cols.target]].max()+1)

		ids = data[:,cols.source] * self.num_nodes + data[:,cols.target]

		idx = data[:,[cols.source,
				   	  cols.target,
				   	  cols.time]]

		self.max_time = data[:,cols.time].max()
		self.min_time = data[:,cols.time].min()

		return {'idx': idx, 'vals': torch.ones(idx.size(0))}


