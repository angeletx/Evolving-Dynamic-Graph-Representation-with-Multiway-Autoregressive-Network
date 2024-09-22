import argparse
import yaml
import torch
import numpy as np
import time
import random
import math

def pad_with_last_col(matrix,cols):
    out = [matrix]
    pad = [matrix[:,[-1]]] * (cols - matrix.size(1))
    out.extend(pad)
    return torch.cat(out,dim=1)

def pad_with_last_val(vect,k):
    device = 'cuda' if vect.is_cuda else 'cpu'
    pad = torch.ones(k - vect.size(0),
                         dtype=torch.long,
                         device = device) * vect[-1]
    vect = torch.cat([vect,pad])
    return vect



def sparse_prepare_tensor(tensor,torch_size, ignore_batch_dim = True):
    if ignore_batch_dim:
        tensor = sp_ignore_batch_dim(tensor)
    tensor = make_sparse_tensor(tensor,
                                tensor_type = 'float',
                                torch_size = torch_size)
    return tensor

def sp_ignore_batch_dim(tensor_dict):
    tensor_dict['idx'] = tensor_dict['idx'][0]
    tensor_dict['vals'] = tensor_dict['vals'][0]
    return tensor_dict

def aggregate_by_time(time_vector,time_win_aggr):
        time_vector = time_vector - time_vector.min()
        time_vector = time_vector // time_win_aggr
        return time_vector

def sort_by_time(data,time_col):
        _, sort = torch.sort(data[:,time_col])
        data = data[sort]
        return data

def print_sp_tensor(sp_tensor,size):
    print(torch.sparse.FloatTensor(sp_tensor['idx'].t(),sp_tensor['vals'],torch.Size([size,size])).to_dense())

def reset_param(t):
    stdv = 2. / math.sqrt(t.size(0))
    t.data.uniform_(-stdv,stdv)

def make_sparse_tensor(adj,tensor_type,torch_size):
    if len(torch_size) == 2:
        tensor_size = torch.Size(torch_size)
    elif len(torch_size) == 1:
        tensor_size = torch.Size(torch_size*2)

    if tensor_type == 'float':
        test = torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
        return torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
    elif tensor_type == 'long':
        return torch.sparse.LongTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.long),
                                      tensor_size)
    else:
        raise NotImplementedError('only make floats or long sparse tensors')

def sp_to_dict(sp_tensor):
    return  {'idx': sp_tensor._indices().t(),
             'vals': sp_tensor._values()}

class Namespace(object):
    '''
    helps referencing object in a dictionary as dict.key instead of dict['key']
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)

def set_seeds(rank):
    seed = int(time.time())+rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

def load_data(file, starting_line=1, sep=',', type_fn = float, tensor_const = torch.DoubleTensor):
    with open(file) as file:
        file = file.read().splitlines()
    data = [[type_fn(r) for r in row.split(sep)] for row in file[starting_line:]]
    data = tensor_const(data)
    return data

def load_data_from_tar(file, tar_archive, replace_unknow=False, starting_line=1, sep=',', type_fn = float, tensor_const = torch.DoubleTensor):
    f = tar_archive.extractfile(file)
    lines = f.read()#
    lines=lines.decode('utf-8')
    if replace_unknow:
        lines=lines.replace('unknow', '-1')
        lines=lines.replace('-1n', '-1')

    lines=lines.splitlines()

    data = [[type_fn(r) for r in row.split(sep)] for row in lines[starting_line:]]
    data = tensor_const(data)
    return data

def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config_file',default='experiments/parameters_example.yaml', type=argparse.FileType(mode='r'), help='optional, yaml file containing parameters to be used, overrides command line parameters')
    return parser

def parse_args(parser):
    args = parser.parse_args()
    if args.config_file:
        data = yaml.load(args.config_file)
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in data.items():
            arg_dict[key] = value

    return args

def get_snapshot_density(data):
    cols = Namespace({'source': 0,
						'target': 1,
						'time': 2})
    _, data[:,[cols.source,cols.target]] = data[:,[cols.source,cols.target]].unique(return_inverse = True)
    num_nodes = int(data[:,[cols.source,cols.target]].max()+1)
    snapshot_density = []
    unique_time = np.unique(data[:,cols.time])

    for t in unique_time:
        indices = data[:,cols.time] == t
        snapshot = data[indices,:]
        ids = snapshot[:,cols.source] * num_nodes + snapshot[:,cols.target]
        num_unique_edges = len(np.unique(ids))
        density = num_unique_edges/num_nodes/(num_nodes-1)
        snapshot_density.append(density)

    density = np.mean(snapshot_density)
    return density