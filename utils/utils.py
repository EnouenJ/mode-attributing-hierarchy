import torch
import numpy as np


from itertools import chain, combinations
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
def oneless(iterable): #subsets one element less
    s = list(iterable)
    return combinations(s, len(s)-1)



def KL(p,q):
#     return torch.sum(p*torch.log(p/q))
    return torch.sum(p*   (torch.log(p)-torch.log(q)))
def KL_plus(p,q):
    return torch.sum(  p[p!=0] * torch.log(  (p/q)[p!=0] )  )


def KL_of_trusted_model(xp,theta_model):
    if theta_model.X_or_P_mode=="X":
        return KL_of_trusted_model_X(xp,theta_model)
    if theta_model.X_or_P_mode=="P":
        return KL_of_trusted_model_P(xp,theta_model)
        
def KL_of_trusted_model_X(x,theta_model):
    #print('x',x.shape)
    with torch.no_grad():
        u,c=torch.unique(x,dim=0,return_counts =True)
        size = x.shape[0]
        c=c.float()
        p_ent = np.log(size) - torch.sum(c*torch.log(c)/size) 

        log_q = theta_model.forward_log_batch(u)

        LSE = 0.0 #baby, the trust
        cross_ent = LSE - torch.sum(c/size*log_q)
    return cross_ent - p_ent

def KL_of_trusted_model_P(p,theta_model):
    #print('p',p.shape)
    with torch.no_grad():
        p_ent = -torch.sum(  p[p!=0] * torch.log(  p[p!=0] )  )
        log_q = theta_model.full_trusted_forward()
        cross_ent = -torch.sum(  p[p!=0]*log_q[p!=0]  )
    return cross_ent - p_ent


def JS_of_empirical_distributions_X(x,w):
        size1 = x.shape[0]
        size2 = w.shape[0]    
        u1,inds1,c1=torch.unique(x,dim=0,sorted=True,return_inverse=True,return_counts=True)
        u2,inds2,c2=torch.unique(w,dim=0,sorted=True,return_inverse=True,return_counts=True)
        
        xw=torch.cat([x,w],dim=0)
        wx=torch.cat([w,x],dim=0)
        u3,inds3,c3=torch.unique(xw,dim=0,sorted=True,return_inverse=True,return_counts=True)
        u4,inds4,c4=torch.unique(wx,dim=0,sorted=True,return_inverse=True,return_counts=True)
        
        p1 = c1[inds1].float()/size1
        p13 = (c3[inds3[:size1]]-c1[inds1]).float()/size2
        #print('p1',p1.shape)
        #print('p13',p13.shape)
        KL1 = torch.mean( torch.log(p1/(0.5*p1+0.5*p13)) )
        #print("KL1",KL1)
        
        p2 = c2[inds2].float()/size2
        p24 = (c4[inds4[:size2]]-c2[inds2]).float()/size1
        #print('p2',p2.shape)
        #print('p24',p24.shape)
        KL2 = torch.mean( torch.log(p2/(0.5*p2+0.5*p24)) )
        #print("KL2",KL2)
        
        JS = 0.5*KL1 + 0.5*KL2
        #print(KL1,KL2,JS)
        #print("JS",JS)
        return JS
#         return KL1
    
    
    

#TODO: add accuracy of X with a few specific classes


# def crossEnt_across_all_directions_P(p,theta_model):
#     pass
#     with torch.no_grad():
#         log_q = theta_model.full_trusted_forward()
#         q = torch.exp(log_q)
#         ce_list = []
        
#         D=len(p.shape)
#         for direc in range(D):
#             pass
        
# #             _,max_i = torch.max(log_q,dim=direc)
# #             sum_i = torch.sum(torch.exp(log_q),dim=direc)
#             sum_i = torch.sum(q,dim=direc,keepdim=True)
    
#             soft_q = q.clone() / sum_i   ##sum_i[ tuple([slice(None)]*(direc-1)+[None]) ]
                        
#             ce_i = -p*torch.log(soft_q)
#             ce_i = torch.sum(ce_i)
#             ce_list.append(float(ce_i.cpu()))
#     return ce_list

def baseEnt_across_mushroom_directions_X(x,I_ks,classes_to_predict):
    cum_ns = []
    cum_n = 0
    for k,I_k in enumerate(I_ks):
        cum_ns.append(cum_n)
        cum_n += I_k
    cum_ns.append(cum_n)
    
    
    base_list = []
    ent_list = []
    for class_idx in classes_to_predict:
        cum_c = cum_ns[class_idx]
        I_c = I_ks[class_idx]
        x_c = x[:, cum_c:cum_c+I_c]
        
        u,c=torch.unique(x_c,dim=0,return_counts=True)
        size = x.shape[0]
        c=c.float()
        clogc = c*torch.log(c)/size
        p_ent = np.log(size) - torch.sum(c*torch.log(c)/size)
        
        ent_list.append(p_ent)
        base_list.append(np.log(I_c))
    return np.array(base_list),np.array(ent_list)
    

def crossEnt_across_mushroom_directions_X(x,theta_model,classes_to_predict):
    I_ks = theta_model.I_ks
    cum_ns = []
    cum_n = 0
    for k,I_k in enumerate(I_ks):
        cum_ns.append(cum_n)
        cum_n += I_k
    cum_ns.append(cum_n)
    
    ce_list = []
    for class_idx in classes_to_predict:
        cum_c = cum_ns[class_idx]
        I_c = I_ks[class_idx]
        x_c = x[:, cum_c:cum_c+I_c]
        
        log_pred_c = torch.log( theta_model.cond_pred_forward(x,class_idx) )
        #print(torch.any(torch.isnan(log_pred_c)))
        #print(torch.any(-float('inf')==(log_pred_c)))
        log_pred_c[torch.isnan(log_pred_c)] = 0    #IS this too generous to the model? i.e. give credit when none is due
        #if True:
        #    print( log_pred_c[torch.any(-float('inf')==(log_pred_c),dim=-1)] )
        #    print( x_c[torch.any(-float('inf')==(log_pred_c),dim=-1)] )
        #    #print(torch.where())
        log_pred_c[(-float('inf')==(log_pred_c))] = 0   #turns out it was too genersou....
        
        log_pred_c = torch.sum(x_c*log_pred_c,axis=-1) #dot with the one-hot encoding
        
        #ce_i = -p*torch.log(soft_q)
        ce_c = -torch.mean( log_pred_c ) #X_mean is same as product with P
        #print(torch.any(torch.isnan(ce_c)))
        #print()
        ce_list.append(float(ce_c.cpu()))
    return ce_list

def Acc_across_mushroom_directions_X(x,theta_model,classes_to_predict):
    I_ks = theta_model.I_ks
    cum_ns = []
    cum_n = 0
    for k,I_k in enumerate(I_ks):
        cum_ns.append(cum_n)
        cum_n += I_k
    cum_ns.append(cum_n)
    
    acc_list = []
    for class_idx in classes_to_predict:
        cum_c = cum_ns[class_idx]
        I_c = I_ks[class_idx]
        x_c = x[:, cum_c:cum_c+I_c]
        
        log_pred_c = theta_model.cond_pred_forward(x,class_idx)
        _,cls_pred = torch.max(log_pred_c,dim=-1)
#         print('cls_pred',cls_pred.shape)
        ###pred_acc = x_c[cls_pred]
        pred_acc = x_c[np.arange(x_c.shape[0]),cls_pred]
#         print('pred_acc',pred_acc.shape)
        
        acc_c = torch.mean( pred_acc )
        acc_list.append(float(acc_c.cpu()))
    return acc_list




from itertools import chain, combinations
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

import datetime
def gettimestamp():
    now = datetime.datetime.now()
    nowstr = now.strftime("%Y%m%d_%H%M%S")
    return nowstr

def param_count(I_ks,tup):
    params = 1
    for i in tup:
        params*=I_ks[i]
    return params