import utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import pprint

class CEGCN(torch.nn.Module):
    def __init__(self, args, activation, device='cpu', skipfeats=False):
        super().__init__()
        common_args = u.Namespace({'num_seq': args.num_seq,
                                  'device': device,
                                  'activation': activation})

        common_args.l1_weight = args.l1_weight
        common_args.l2_weight = args.l2_weight

        if isinstance(args.hidden_feats, int):
            args.hidden_feats = [args.hidden_feats]
        feats = [args.feats_per_node] + args.hidden_feats

        self.device = device
        self.skipfeats = skipfeats
        self.model = []
        self._parameters = nn.ParameterList()
        self.reg_param_list = []

        CEGCN_args = common_args
        for i in range(1,len(feats)):
            CEGCN_args.in_feats = feats[i-1]
            CEGCN_args.out_feats = feats[i]

            cegcn_i = CEGCN_layer(CEGCN_args)
            self.model.append(cegcn_i.to(self.device))
            self._parameters.extend(list(self.model[i-1].parameters()))
            self.reg_param_list.extend(cegcn_i.get_reg_param_list())

    def parameters(self):
        return self._parameters

    def forward(self,A_list, Nodes_list,nodes_mask_list):

        for i in range(0,len(self.model)):
            layer_i = self.model[i]
            Nodes_list = layer_i(A_list,Nodes_list)#,nodes_mask_list)

        out = Nodes_list[-1]
        return out


class CEGCN_layer(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args

        self.activation = args.activation
        
        self.evolve_weights_vert = mat_GRU_cell(args.in_feats,args.out_feats)
        
        self.GCN_vert_init_weights = Parameter(torch.Tensor(args.in_feats,args.out_feats))
        self.reset_param(self.GCN_vert_init_weights)
        
        self.net_gcn_vert = torch.nn.Linear(args.out_feats,args.out_feats) 
        self.net_pt_l = torch.nn.Linear(args.out_feats,args.out_feats)
        self.net_pt_pl = torch.nn.Linear(args.in_feats,args.out_feats)
        self.net_t_pl = torch.nn.Linear(args.in_feats,args.out_feats)

    def reset_param(self,t):
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def get_reg_param_list(self):
        regParamList = []
        regParamList.extend(self.net_gcn_vert.parameters())
        regParamList.extend(self.net_t_pl.parameters())
        regParamList.extend(self.net_pt_pl.parameters())
        regParamList.extend(self.net_pt_l.parameters())
        return regParamList

    def forward(self,A_list,node_embs_list):#,mask_list):
        W_v = self.GCN_vert_init_weights
        
        node_emb_current_layer = []

        for t,Ahat in enumerate(A_list):
            node_embs_t_pl = node_embs_list[t]
        
            if t > 0:
                node_embs_pt_pl = node_embs_list[t-1]
                node_embs_pt_l = node_emb_current_layer[-1]
            else:
                node_embs_pt_pl = torch.zeros_like(node_embs_t_pl).to(self.args.device)
                node_embs_pt_l = torch.zeros(node_embs_t_pl.size(0), self.args.out_feats).to(self.args.device)

            W_v = self.evolve_weights_vert(W_v)
  
            node_embs_gcn_vert = self.activation(Ahat.matmul(node_embs_t_pl.matmul(W_v)))

            node_embs = self.net_t_pl(node_embs_t_pl) + self.net_gcn_vert(node_embs_gcn_vert) + self.net_pt_l(node_embs_pt_l) + self.net_pt_pl(node_embs_pt_pl)
            
            node_embs = self.activation(node_embs)
            node_emb_current_layer.append(node_embs)

        return node_emb_current_layer

class mat_GRU_cell(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.update = mat_GRU_gate(in_feats,
                                   out_feats,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(in_feats,
                                   out_feats,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(in_feats,
                                   out_feats,
                                   torch.nn.Tanh())

    def forward(self,prev_Q):#,prev_Z,mask):

        z_topk = prev_Q

        update = self.update(z_topk,prev_Q)
        reset = self.reset(z_topk,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q
        
class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

