#from turtle import end_fill
import torch
import utils as u

class Cross_Entropy(torch.nn.Module):
    """docstring for Cross_Entropy"""
    def __init__(self, args, dataset):
        super().__init__()
        weights = torch.tensor(args.class_weights).to(args.device)

        self.weights = self.dyn_scale(args.task, dataset, weights)
        
    
    def dyn_scale(self,task,dataset,weights):
        def scale(labels):
            return weights
        return scale
    

    def logsumexp(self,logits):
        m,_ = torch.max(logits,dim=1)
        m = m.view(-1,1)
        sum_exp = torch.sum(torch.exp(logits-m),dim=1, keepdim=True)
        return m + torch.log(sum_exp)
    
    def forward(self,logits,labels,parameterList=[],l1_weight=[],l2_weight=[]):
        '''
        logits is a matrix M by C where m is the number of classifications and C are the number of classes
        labels is a integer tensor of size M where each element corresponds to the class that prediction i
        should be matching to
        '''
        labels = labels.view(-1,1)
        alpha = self.weights(labels)[labels].view(-1,1)
        loss = alpha * (- logits.gather(-1,labels) + self.logsumexp(logits))

        loss = loss.mean()
        listLength = len(parameterList)
        
        if listLength:
            l1_reg_loss = 0
            l2_reg_loss = 0
            num_l1 = 1
            num_l2 = 1

            if len(l1_weight) > 0:
                num_l1 = 0
                for i,p in enumerate(parameterList):
                    index = int(i/2) % 4

                    if l1_weight[index] > 0:
                        addition_l1 = torch.abs(p).sum() * l1_weight[index]
                        l1_reg_loss += addition_l1
                        if addition_l1 > 0:
                            num_l1 += p.numel()

            if len(l2_weight) > 0:
                num_l2 = 0
                for i,p in enumerate(parameterList):
                    index = int(i/2) % 4

                    if l2_weight[index] > 0:
                        addition_l2 = p.pow(2.0).sum() * l2_weight[index]
                        l2_reg_loss += addition_l2
                        if addition_l2 > 0:
                            num_l2 += p.numel() 

            if num_l1 > 0:
                loss = loss + l1_reg_loss/num_l1
            if num_l2 > 0:
                loss = loss + l2_reg_loss/num_l2

        return loss


