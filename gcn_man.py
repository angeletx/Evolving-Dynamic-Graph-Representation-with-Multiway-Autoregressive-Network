import utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import pprint

class CGCN(torch.nn.Module):
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
        self.model = []
        self._parameters = nn.ParameterList()
        self.reg_param_list = []

        CGCN_args = common_args
        for i in range(1,len(feats)):
            CGCN_args.in_feats = feats[i-1]
            CGCN_args.out_feats = feats[i]

            cgcn_i = CGCN_layer(CGCN_args)
            self.model.append(cgcn_i.to(self.device))
            self._parameters.extend(list(self.model[i-1].parameters()))
            self.reg_param_list.extend(cgcn_i.get_reg_param_list())

    def parameters(self):
        return self._parameters

    def forward(self,A_list, Nodes_list,nodes_mask_list):

        for i in range(0,len(self.model)):
            layer_i = self.model[i]
            Nodes_list = layer_i(A_list,Nodes_list)#,nodes_mask_list)

        out = Nodes_list[-1]
        return out


class CGCN_layer(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.activation = self.args.activation

        self.W_v = Parameter(torch.Tensor(args.in_feats,args.out_feats))
        
        self.net_gcn_vert = torch.nn.Linear(args.out_feats,args.out_feats) 
        self.net_pt_l = torch.nn.Linear(args.out_feats,args.out_feats)
        self.net_pt_pl = torch.nn.Linear(args.in_feats,args.out_feats)
        self.net_t_pl = torch.nn.Linear(args.in_feats,args.out_feats)
        
        self.reset_param(self.W_v)

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

        node_emb_current_layer = []

        for t,Ahat in enumerate(A_list):
            node_embs_t_pl = node_embs_list[t]
        
            if t > 0:
                node_embs_pt_pl = node_embs_list[t-1]
                node_embs_pt_l = node_emb_current_layer[-1]
            else:
                node_embs_pt_pl = torch.zeros_like(node_embs_t_pl).to(self.args.device)
                node_embs_pt_l = torch.zeros(node_embs_t_pl.size(0), self.args.out_feats).to(self.args.device)            

            node_embs_gcn_vert = self.activation(Ahat.matmul(node_embs_t_pl.matmul(self.W_v)))
            
            node_embs = self.net_t_pl(node_embs_t_pl) + self.net_gcn_vert(node_embs_gcn_vert) + self.net_pt_l(node_embs_pt_l) + self.net_pt_pl(node_embs_pt_pl)
            
            node_embs = self.activation(node_embs)
            node_emb_current_layer.append(node_embs)

        return node_emb_current_layer
