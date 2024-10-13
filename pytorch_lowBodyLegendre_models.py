import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import math
from itertools import chain, combinations
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    






class LowBodyLegendre_LogLinearGAM(nn.Module):
    def __init__(self, my_inters, I_ks, constant_version=None, X_or_P_mode="X"):
        super().__init__()


        self.I_ks = I_ks
        self.my_inters = my_inters
        self.cum_I_ks = []
        self.div_I_ks = []
        D=len(self.I_ks)
        cumsum=0
        for i in range(D):
            I_i = self.I_ks[i]
            self.cum_I_ks.append(cumsum)
            cumsum += I_i 
        self.cum_I_ks.append(cumsum) #still add full sum to the end
        cumprod=1
        for i in (range(D)):
            I_i = self.I_ks[i]
            self.div_I_ks.append(cumprod)
            cumprod *= I_i
        self.div_I_ks.append(cumprod) #still add full product to the end



        self.theta_parameters = nn.ParameterDict()
        self.theta_masks = None
        self.bad_tens_dict = None
        self.split_theta_naughts = None

        self.constant_shift = 0
        self.all_good_idx_tensor_list = None
        self.my_hemi_uniform_dict = None

        self.block_building_params = None
        self.new_tups = []


        self.X_or_P_mode = X_or_P_mode


        if constant_version is None:
            constant_version = "UNIT_CONST"
        self.constant_version = constant_version

        if constant_version=="CONJ_SQRT_CONST":
            global_const = 1
            for i in range(len(I_ks)):
                global_const*=I_ks[i]
                print("global_const",global_const)
                global_const=np.sqrt(global_const)
        self.const_parameters = {}
        for tt,tup in enumerate(my_inters):
            theta_shape_tt = tuple([I_ks[i] for i in tup])
            self.theta_parameters[str(tup)] = nn.Parameter( torch.zeros( theta_shape_tt ) )
            if constant_version=="SQRT_CONST": #SQRT version
                self.const_parameters[str(tup)] = float(np.sqrt(  torch.zeros( theta_shape_tt ).reshape(-1).shape[0] ))
            if constant_version=="INV_SQRT_CONST": #INV SQRT version
                self.const_parameters[str(tup)] = 1/ float(np.sqrt(  torch.zeros( theta_shape_tt ).reshape(-1).shape[0] ))
            if constant_version=="CONJ_SQRT_CONST": #reverse SQRT version (const * INV)
                self.const_parameters[str(tup)] = global_const/ float(np.sqrt(  torch.zeros( theta_shape_tt ).reshape(-1).shape[0] ))
            if constant_version=="UNIT_CONST":  #UNIT  version
                self.const_parameters[str(tup)] = 1
            if constant_version=="INV_CONST": #INV  version
                self.const_parameters[str(tup)] = 1/ float((  torch.zeros( theta_shape_tt ).reshape(-1).shape[0] ))


                


        

    def initialize_from_numpy_thetas(self, theta_dict):        
        self.theta_parameters = nn.ParameterDict()
        for tt,tup in enumerate(self.my_inters):
            self.theta_parameters[str(tup)] = nn.Parameter( torch.Tensor( theta_dict[tup] )/self.const_parameters[str(tup)] )
            



    def add_new_interaction_tuple(self, tup, x_full, recompute_masks=True, recompute_hemis=True):
        pass

        #UPDATE THE THETAS
        curr_device = self.theta_parameters["()"].device
        self.my_inters.append( tup )
        theta_shape_tt = tuple([self.I_ks[i] for i in tup])
        #print('\t',theta_shape_tt )
        self.theta_parameters[str(tup)] = nn.Parameter( torch.zeros( theta_shape_tt ).to(curr_device) )
        if self.constant_version=="UNIT_CONST":  #UNIT  version
            self.const_parameters[str(tup)] = 1
        else:
            print("NOT IMPMLEMENTED YET")

        #ADD THE THETA MASKS
        if recompute_masks:
            self.update_new_masks_to_thetas([tup],x_full)

        #UPDATE THE VALID HEMIS BY REGROUPING THE MAX_SPLITTING
        if recompute_hemis:
            self.update_all_valid_split_hemi_indices(tup)

        #UPDATE THE HEMIS ACCORDING TO THE CORRECTED HEMI_VALID_INDS
        if recompute_hemis:
            self.precompute_all_split_hemi_taus()


        #verbose
        if recompute_hemis:
            for s,split in enumerate(self.maximal_splitting):
                all_good_idx_tensor = self.all_good_idx_tensor_list[s]
                print(split,'\t',all_good_idx_tensor.shape)


    def update_new_masks_to_thetas(self, new_tups, xp_full):
        eta_dict = self.get_eta_vec_tensor(xp_full)
        curr_device = xp_full.device
        # print('updating')
        # print('eta_dict',eta_dict)

        for tup in new_tups:
            eta_tt = eta_dict[tup]
            theta_mask_tt = (eta_tt==0).float()

            wrapped_up_arange = torch.arange(theta_mask_tt.reshape(-1).shape[0],device=eta_tt.device).reshape(theta_mask_tt.shape)
            bad_indices = wrapped_up_arange[eta_tt==0].reshape(-1)
            if len(bad_indices)>0:
                bad_1D_tensors = []
                cumprod=1
                idx = bad_indices
                for i in reversed(tup): #reversed() corresponds w/ the default behavior of reshape(-1)
                    I_i = self.I_ks[i]
                    cumprod *= I_i

                    Z_i_flat = (idx % I_i)
                    idx = idx // I_i

                    Z_i = torch.zeros( (idx.shape[0],I_i), device=curr_device).scatter_(1, Z_i_flat[:,None], 1.)
                    bad_1D_tensors.append( Z_i )
                bad_tensors = torch.cat(list(reversed(bad_1D_tensors)),dim=-1)
            else:
                bad_tensors = torch.zeros([0]).to(curr_device) 
                
                
            self.theta_masks[str(tup)] = nn.Parameter(theta_mask_tt,requires_grad=False).to(curr_device)
            self.bad_tens_dict[tup] = bad_tensors
        pass

    def update_all_valid_split_hemi_indices(self, tup):

        if len(tup)>1:
            tup_split_inds = self.group_splittings_according_to_tup(self.maximal_splitting,tup)
            zbs = 10*1000 #magic number
            curr_device = self.theta_parameters["()"].device

            if len(tup_split_inds)>1:
                print("OKAY, WE NEED TO GROUP:",tup_split_inds)
                if len(tup_split_inds)>2:
                    print("SCREAM BECAUSE A TRIPLE BROUGHT A MAX SPLIT")
                tup_split_inds=sorted(tup_split_inds,reverse=True)
                popped_splits = [self.maximal_splitting.pop(s1) for s1 in tup_split_inds]
                popped_good_indices = [self.all_good_idx_tensor_list.pop(s1) for s1 in tup_split_inds]
                
                new_split = list(chain(*popped_splits))
                new_split = sorted(new_split) 
                self.maximal_splitting.append( new_split )

                #create the new good indices

                new_theta_list = nn.ParameterList()
                for ss in range(len(self.maximal_splitting)):
                    new_theta_list.append( nn.Parameter( torch.zeros( () ).to(curr_device) ) )
                self.split_theta_naughts = new_theta_list


                for s,split in enumerate(self.maximal_splitting[-1:]):
                    s=len(self.maximal_splitting)-1 #only update last one
                    split_tup = tuple(split)
                    split_dict = {}
                    for ii,i in enumerate(split_tup):
                        split_dict[i] = ii
                    print('split_tup',split_tup)
                    print('s',s)

                    curr_bs = zbs

                    cand_good_idx_tensor = torch.zeros(1).long()
                    full_cumprod = 1
                    g_dims = []
                    for ss,split2 in enumerate(popped_splits):
                        exp_tup = tuple([None]*(ss+0))
                        good_ss = popped_good_indices[ss]
                        cand_good_idx_tensor = cand_good_idx_tensor + good_ss[exp_tup]
                        g_ss = good_ss.shape[0]
                        g_dims.append(g_ss)
                        full_cumprod *= g_ss
                        cand_good_idx_tensor = cand_good_idx_tensor.reshape(tuple(g_dims+[1]))
                        pass
                    cand_good_idx_tensor = cand_good_idx_tensor.reshape(-1)
                    

                    all_good_idxes = []
                    for zb in range( int(np.ceil(full_cumprod/curr_bs)) ):
                        if (zb+1)*zbs>full_cumprod:
                            curr_bs = full_cumprod % zbs
                        
                        idx_global = cand_good_idx_tensor[zb*zbs:zb*zbs+curr_bs].to(curr_device)
                        split_onedim_tensors = self.create_split_onedim_tensors_from_global_indices(idx_global,curr_bs,curr_device,s) 
                        if self.theta_masks is not None:
                            bad_hits = self.split_validate_against_peaks(split_onedim_tensors,s,split_dict)
                            good_idx = idx_global[~bad_hits.cpu()]
                        else:
                            good_idx = idx_global
                        
                        all_good_idxes.append(good_idx.cpu())
                    all_good_idx_tensor = torch.cat(all_good_idxes,dim=-1)
                    print(split,'\t',all_good_idx_tensor.shape)
                    self.all_good_idx_tensor_list.append( all_good_idx_tensor )
            else:
                curr_s = tup_split_inds[0]
                curr_split = self.maximal_splitting[curr_s]

                split_tup = tuple(curr_split)
                split_dict = {}
                for ii,i in enumerate(split_tup):
                    split_dict[i] = ii

                
                if True:
                    all_good_idxes = []
                    cand_good_idx_tensor = self.all_good_idx_tensor_list[curr_s]
                    full_cumprod = cand_good_idx_tensor.shape[0]
                    curr_bs = zbs
                    for zb in range( int(np.ceil(full_cumprod/curr_bs)) ):
                        if (zb+1)*zbs>full_cumprod:
                            curr_bs = full_cumprod % zbs
                        
                        idx_global = cand_good_idx_tensor[zb*zbs:zb*zbs+curr_bs].to(curr_device)
                        split_onedim_tensors = self.create_split_onedim_tensors_from_global_indices(idx_global,curr_bs,curr_device,curr_s) 
                        if self.theta_masks is not None:
                            bad_hits = self.split_validate_against_peaks(split_onedim_tensors,curr_s,split_dict)
                            good_idx = idx_global[~bad_hits.cpu()]
                        else:
                            good_idx = idx_global
                        

                        all_good_idxes.append(good_idx.cpu())
                    all_good_idx_tensor = torch.cat(all_good_idxes,dim=-1)
                    
                    self.all_good_idx_tensor_list[curr_s] = ( all_good_idx_tensor )

                pass
        pass



    def add_the_masks_to_thetas(self, xp_full):
        self.theta_masks = nn.ParameterDict()
        self.bad_tens_dict = {}
        eta_dict = self.get_eta_vec_tensor(xp_full)
        curr_device = xp_full.device

        for tt,tup in enumerate(self.my_inters):
            eta_tt = eta_dict[tup]
            theta_mask_tt = (eta_tt==0).float()

            wrapped_up_arange = torch.arange(theta_mask_tt.reshape(-1).shape[0],device=eta_tt.device).reshape(theta_mask_tt.shape)
            bad_indices = wrapped_up_arange[eta_tt==0].reshape(-1)
            if len(bad_indices)>0:
                bad_1D_tensors = []
                cumprod=1
                idx = bad_indices

                for i in reversed(tup): #reversed() corresponds w/ the default behavior of reshape(-1)
                    I_i = self.I_ks[i]
                    cumprod *= I_i

                    Z_i_flat = (idx % I_i)
                    idx = idx // I_i

                    Z_i = torch.zeros( (idx.shape[0],I_i), device=curr_device).scatter_(1, Z_i_flat[:,None], 1.)
                    bad_1D_tensors.append( Z_i )
                bad_tensors = torch.cat(list(reversed(bad_1D_tensors)),dim=-1)
            else:
                bad_tensors = torch.zeros([0]).to(curr_device) 
                
            self.theta_masks[str(tup)] = nn.Parameter(theta_mask_tt,requires_grad=False).to(curr_device)
            self.bad_tens_dict[tup] = bad_tensors
        pass


    def get_eta_vec_tensor(self, xp, inters=None):
        if self.X_or_P_mode=="X":
            return self.get_eta_vec_tensor_X(xp,inters)
        if self.X_or_P_mode=="P":
            return self.get_eta_vec_tensor_safer_logP(torch.log(xp),inters)

    def get_eta_vec_tensor_X(self, x, inters=None):
        etas_dict = {}
        if inters is None:
            inters = self.my_inters
        
        curr_bs = x.shape[0]
        curr_device = x.device
        onedim_tensors = self.split_apart_x_batch(x)
        for tt,tup in enumerate(inters):
            cumprod=1
            X_tt = torch.ones( (curr_bs,cumprod),device=curr_device )
            for i in tup:
                X_tt = torch.einsum("ij,ik->ijk",X_tt, onedim_tensors[i])
                cumprod *= self.I_ks[i]
                X_tt = X_tt.reshape( (-1,cumprod) )
            X_tt = X_tt.reshape( tuple([-1]+[self.I_ks[i] for i in tup]) )
            etas_dict[tup] = torch.mean(X_tt,dim=0)
        return etas_dict
    
    def get_eta_vec_tensor_P(self, p, inters=None):
        etas_dict = {}
        if inters is None:
            inters = self.my_inters
        
        curr_bs = p.shape[0]
        curr_device = p.device
        D = len(self.I_ks)
        for tt,tup in enumerate(inters):
            axes_to_sum = []
            for i in range(D):
                if i not in tup:
                    axes_to_sum.append(i)
                    
            if len(axes_to_sum)>0:
                etas_dict[tup] = torch.sum(p,dim=tuple(axes_to_sum))
            else:
                etas_dict[tup] = p
        return etas_dict

    def get_eta_vec_tensor_safer_logP(self, log_p, inters=None):
        etas_dict = {}
        if inters is None:
            inters = self.my_inters
        
        curr_device = log_p.device
        D = len(self.I_ks)
        for tt,tup in enumerate(inters):
            axes_to_sum = []
            for i in range(D):
                if i not in tup:
                    axes_to_sum.append(i)
                    
            if True:
                axes_to_sum = list(reversed(axes_to_sum)) 
            if len(axes_to_sum)>0:
                log_eta_tt = torch.logsumexp(log_p,dim=tuple(axes_to_sum))
            else:
                log_eta_tt = log_p
            etas_dict[tup] = torch.exp(log_eta_tt)
        return etas_dict

        
    def compute_pseudolikelihood_custom_backwards_split_flattened_etas_X(self, x):


        with torch.no_grad():
            curr_device = x.device
            
            target_eta_dict = {}
            for tt,tup in enumerate(self.my_inters):
                target_eta_dict[tup] = torch.zeros_like(  self.theta_parameters[str(tup)], device=curr_device  )
            if True: #no batches needed for dataset right now
                pass
                curr_bs=x.shape[0]
            
                for tt,tup in enumerate(self.my_inters):
                    X_tt = self.fast_eta_reshaped_batching(x,tup,curr_device)
                    
                    target_eta_dict[tup] += torch.sum(X_tt,dim=0)
            for tt,tup in enumerate(self.my_inters):
                target_eta_dict[tup] = target_eta_dict[tup] / torch.sum(target_eta_dict[tup])
                target_eta_dict[tup] = target_eta_dict[tup] * len(tup) #whoops! forgot this at first





            current_eta_dict = {}
            onedim_tensors = self.split_apart_x_batch(x)
            for tt,tup in enumerate(self.my_inters):
                current_eta_dict[tup] = torch.zeros_like(  self.theta_parameters[str(tup)], device=curr_device  )

            D = len(self.I_ks)
            for d in range(D):
                
                prob_d = self.cond_pred_forward(x,d) 
                print(d,'prob_d',prob_d.shape)
                for xx in range(3):
                    print('\t',prob_d[xx])

                for tt,tup in enumerate(self.my_inters):
                    if d in tup:
                        cumprod=1
                        C_tt = torch.ones( (curr_bs,cumprod),device=curr_device )
                        for i in tup:
                            if i!=d:
                                C_tt = torch.einsum("ij,ik->ijk",C_tt, onedim_tensors[i])
                            else:
                                C_tt = torch.einsum("ij,ik->ijk",C_tt, prob_d)
                            cumprod *= self.I_ks[i]
                            C_tt = C_tt.reshape( (-1,cumprod) )
                        C_tt = C_tt.reshape( tuple([-1]+[self.I_ks[i] for i in tup]) )
                        #finally accumulate for this particular 'd'
                        current_eta_dict[tup] += torch.mean(C_tt,dim=0)
            self.current_eta_dict = current_eta_dict




            gradient_dict = {}
            for tt,tup in enumerate(self.my_inters):
                gradient_dict[tup] = -target_eta_dict[tup] + current_eta_dict[tup]
                
            for tup in gradient_dict:
                self.theta_parameters[str(tup)].grad = gradient_dict[tup].clone()



    def compute_custom_backwards_split_flattened_etas(self, xp):
        if self.X_or_P_mode=="X":
            return self.compute_custom_backwards_split_flattened_etas_X(xp)
        if self.X_or_P_mode=="P":
            return self.compute_custom_backwards_split_flattened_etas_P(xp)


    def compute_custom_backwards_split_flattened_etas_P(self, p):
        with torch.no_grad():
            target_eta_dict = self.get_eta_vec_tensor(p);

            log_q = self.full_trusted_forward()
            current_eta_dict = self.get_eta_vec_tensor_safer_logP( log_q );



            gradient_dict = {}
            for tt,tup in enumerate(self.my_inters):
                gradient_dict[tup] = -target_eta_dict[tup] + current_eta_dict[tup]            
            for tup in gradient_dict:
                self.theta_parameters[str(tup)].grad = gradient_dict[tup].clone()

    def compute_custom_backwards_split_flattened_etas_X(self, x):
        with torch.no_grad():
            curr_device = x.device
            target_eta_dict = {}
            for tt,tup in enumerate(self.my_inters):
                target_eta_dict[tup] = torch.zeros_like(  self.theta_parameters[str(tup)], device=curr_device  )
            if True: #no batches needed for dataset right now
                pass
                curr_bs=x.shape[0]
            
                for tt,tup in enumerate(self.my_inters):
                    X_tt = self.fast_eta_reshaped_batching(x,tup,curr_device)
                    
                    target_eta_dict[tup] += torch.sum(X_tt,dim=0)
            for tt,tup in enumerate(self.my_inters):
                target_eta_dict[tup] = target_eta_dict[tup] / torch.sum(target_eta_dict[tup])




            current_eta_dict = {}
            for tt,tup in enumerate(self.my_inters):
                current_eta_dict[tup] = []

            zbs = 10*1000


            for ss,split in enumerate(self.maximal_splitting):
                split_tup = tuple(split)
                all_good_idx_tensor = self.all_good_idx_tensor_list[ss]
                full_hemi_cumprod = all_good_idx_tensor.shape[0]

                semiglobal_div_integers = []
                semiglobal_mod_integers = [] #self.I_ks
                cumprod=1

                split_dict = {}
                for ii,i in enumerate(split_tup):
                    I_i = self.I_ks[i]
                    semiglobal_div_integers.append( cumprod )
                    semiglobal_mod_integers.append(   I_i   )
                    cumprod *= I_i
                    split_dict[i] = ii
                    
                curr_bs = zbs
                for zb in range( int(np.ceil(full_hemi_cumprod/curr_bs)) ):
                    if (zb+1)*zbs>full_hemi_cumprod:
                        curr_bs = full_hemi_cumprod % zbs
                    
                    idx = all_good_idx_tensor[zb*zbs:zb*zbs+curr_bs].to(curr_device)
                    onedim_tensors = self.create_split_onedim_tensors_from_global_indices(idx,curr_bs,curr_device,ss)
                    theta_Z = self.get_batch_from_split_onedim_tensors(onedim_tensors,curr_bs,curr_device,ss,split_dict)
                    

                    for tt,tup in enumerate(self.my_inters):
                        if tup!=() and tup[0] in split_tup: #only remember those which are relevant to the current split
                            Z_i_flats = [((idx//self.div_I_ks[i]) % self.I_ks[i])   for i in tup] 
                            cumprod_tt = 1
                            for i in tup:
                                cumprod_tt *= self.I_ks[i]
                            cumprods_tt = []
                            local_cumprod_tt = cumprod_tt
                            for i in tup:
                                cumprod_tt //= self.I_ks[i]
                                cumprods_tt.append(cumprod_tt) #this version reverses the order

                            Z_ind_tt_2 = torch.zeros_like(theta_Z,dtype=int)
                            for ii,i in enumerate(tup):
                                Z_ind_tt_2 += cumprods_tt[ii] * Z_i_flats[ii] 

                            if True:  
                                SE_tt = torch.zeros((local_cumprod_tt,),device=curr_device)
                                SE_tt.index_put_( indices=(Z_ind_tt_2,), values=torch.exp(theta_Z), accumulate=True)

                                LSE_tt = torch.log(SE_tt)
                                current_eta_dict[tup].append(LSE_tt)
        

            for tt,tup in enumerate(self.my_inters):
                if tup!=():
                    flat_eta_tt =  torch.exp(  torch.logsumexp(  torch.stack(current_eta_dict[tup]),dim=0  )  )
                    current_eta_dict[tup] = flat_eta_tt.reshape(  tuple([self.I_ks[i] for i in tup]) )
                else:
                    current_eta_dict[tup] = torch.ones(())
            self.current_eta_dict = current_eta_dict

            gradient_dict = {}
            for tt,tup in enumerate(self.my_inters):
                gradient_dict[tup] = -target_eta_dict[tup] + current_eta_dict[tup]
                

            for tup in gradient_dict:
                self.theta_parameters[str(tup)].grad = gradient_dict[tup].clone()



    def fast_eta_reshaped_batching(self, curr_x, curr_tup,curr_device):
        def build_block(my_tup):
            cumprod=1
            cumsum=0
            local_cumsums=[]
            for i in my_tup:
                local_cumsums.append(cumsum)
                cumprod *= self.I_ks[i]
                cumsum  += self.I_ks[i]
            pass

            K = len(my_tup)
            stacked_weights_list = []
            for jj,j in enumerate(my_tup):
                stacked_weights_j = torch.zeros( tuple([self.I_ks[j]]+[self.I_ks[i] for i in my_tup]) )
                for e_j in range(self.I_ks[j]):
                    tup_e_j = tuple([e_j]+[slice(None)]*(jj)+[e_j]+[slice(None)]*(K-jj-1))
                    stacked_weights_j[tup_e_j] += 1 / (K-0.5)
                stacked_weights_list.append(stacked_weights_j)
                pass

            fully_stacked_weights = torch.zeros( tuple([self.cum_I_ks[-1]]+[self.I_ks[i] for i in my_tup]) )
            for jj,j in enumerate(my_tup):
                cum_j = self.cum_I_ks[j]
                I_j = self.I_ks[j]
                fully_stacked_weights[cum_j:cum_j+I_j] = stacked_weights_list[jj]
            fully_stacked_weights = fully_stacked_weights.reshape(-1,cumprod)
            self.block_building_params[str(my_tup)] = nn.Parameter(fully_stacked_weights.to(curr_device),requires_grad=False)
            


        if curr_tup==():
            return torch.ones(curr_x.shape[0],device=curr_device)
        if self.block_building_params is None:
            self.block_building_params = nn.ParameterDict()
        if curr_tup not in self.block_building_params:
            build_block(curr_tup)
            
        #actual logic
        flat_ind_tt = self.block_building_params[str(curr_tup)]
        X_tt = (torch.matmul(curr_x,flat_ind_tt)>1).float()
        X_tt = X_tt.reshape( tuple([-1]+[self.I_ks[i] for i in curr_tup]) )
        return X_tt

    def compute_custom_backwards_split_flattened_etas_utilizing_gibbs_X(self, x, w):
        with torch.no_grad():

            curr_device = x.device


            curr_device = x.device
            onedim_tensors = self.split_apart_x_batch(x) 
            target_eta_dict = {}
            for tt,tup in enumerate(self.my_inters):
                target_eta_dict[tup] = torch.zeros_like(  self.theta_parameters[str(tup)], device=curr_device  )
            if True: #no batches needed for dataset right now
                pass
                curr_bs=x.shape[0]
            
                for tt,tup in enumerate(self.my_inters):
                    cumprod=1
                    X_tt = torch.ones( (curr_bs,cumprod),device=curr_device )
                    for i in tup:
                        X_tt = torch.einsum("ij,ik->ijk",X_tt, onedim_tensors[i])
                        cumprod *= self.I_ks[i]
                        X_tt = X_tt.reshape( (-1,cumprod) )
                    X_tt = X_tt.reshape( tuple([-1]+[self.I_ks[i] for i in tup]) )
                    # X_tt = self.fast_eta_reshaped_batching(x,tup,curr_device)
                    
                    target_eta_dict[tup] += torch.sum(X_tt,dim=0)
            for tt,tup in enumerate(self.my_inters):
                target_eta_dict[tup] = target_eta_dict[tup] / torch.sum(target_eta_dict[tup])



            current_eta_dict = {}
            onedim_tensors_w = self.split_apart_x_batch(w)
            for tt,tup in enumerate(self.my_inters):
                current_eta_dict[tup] = torch.zeros_like(  self.theta_parameters[str(tup)], device=curr_device  )
            if True:
                curr_bs=w.shape[0] #no batches yet
            
                for tt,tup in enumerate(self.my_inters):
                    cumprod=1
                    W_tt = torch.ones( (curr_bs,cumprod),device=curr_device )
                    for i in tup:
                        W_tt = torch.einsum("ij,ik->ijk",W_tt, onedim_tensors_w[i])
                        cumprod *= self.I_ks[i]
                        W_tt = W_tt.reshape( (-1,cumprod) )
                    W_tt = W_tt.reshape( tuple([-1]+[self.I_ks[i] for i in tup]) )
                    
                    current_eta_dict[tup] += torch.sum(W_tt,dim=0)
            for tt,tup in enumerate(self.my_inters):
                current_eta_dict[tup] = current_eta_dict[tup] / torch.sum(current_eta_dict[tup])
            self.current_eta_dict = current_eta_dict




            gradient_dict = {}
            for tt,tup in enumerate(self.my_inters):
                gradient_dict[tup] = -target_eta_dict[tup] + current_eta_dict[tup]
                

            for tup in gradient_dict:
                self.theta_parameters[str(tup)].grad = gradient_dict[tup].clone()


                
    def create_split_onedim_tensors_from_semiglobal_indices(self,idx,curr_bs,curr_device,curr_split):
        split_tup = self.maximal_splitting[curr_split]
        onedim_tensors = []
        cumprod=1
        for i in split_tup:
            I_i = self.I_ks[i]
            cumprod *= I_i

            Z_i_flat = (idx % I_i)
            idx = idx // I_i

            Z_i = torch.zeros( (curr_bs,I_i), device=curr_device).scatter_(1, Z_i_flat[:,None], 1.)
            onedim_tensors.append( Z_i )
        return onedim_tensors
    
    def create_split_onedim_tensors_from_global_indices(self,idx,curr_bs,curr_device,curr_split):
        split_tup = self.maximal_splitting[curr_split]
        onedim_tensors = []
        for i in split_tup:
            I_i = self.I_ks[i]
            Z_i_flat = (idx//self.div_I_ks[i]) % I_i
            
            if False:
                Z_i_flat = (idx % I_i)
                idx = idx // I_i

            Z_i = torch.zeros( (curr_bs,I_i), device=curr_device).scatter_(1, Z_i_flat[:,None], 1.)
            onedim_tensors.append( Z_i )
        return onedim_tensors
    
    def get_batch_from_split_onedim_tensors(self,onedim_tensors,curr_bs,curr_device,curr_split,split_dict):
        theta_X = 0
        split_tup = self.maximal_splitting[curr_split]
        for tt,tup in enumerate(self.my_inters):

            theta_X_tt = None
            if tup!=():
                if tup[0] in split_tup: #assumes max splitting was done properly
                    theta_tt = self.theta_parameters[str(tup)]*self.const_parameters[str(tup)]
                
                    cumprod=1
                    X_tt = torch.ones( (curr_bs,cumprod),device=curr_device )
                    for i in tup:
                        ii=split_dict[i] #index within the subset of onedim tensors (I did not construct all 1D tensors)
                        X_tt = torch.einsum("ij,ik->ijk",X_tt, onedim_tensors[ii])
                        cumprod *= self.I_ks[i]
                        X_tt = X_tt.reshape( (-1,cumprod) )
                    X_tt = X_tt.reshape( tuple([-1]+[self.I_ks[i] for i in tup]) )

                    theta_X_tt = theta_tt[None] * X_tt
                    if tup!=(): #now this correction is in the same place everywhere.
                        axes_to_sum_tt = list(range(1,len(tup)+1))
                        theta_X_tt = torch.sum(theta_X_tt, dim=axes_to_sum_tt)
                        
            else: #tup=(), need to do the customization for splitting
                #theta_tt = self.theta_parameters[str(tup)]*self.const_parameters[str(tup)]
                theta_tt = self.split_theta_naughts[curr_split]*self.const_parameters[str(tup)]
                X_tt = torch.ones( (curr_bs,),device=curr_device )
                theta_X_tt = theta_tt[None] * X_tt


            if theta_X_tt is not None:
                if self.theta_masks is None:
                    theta_X = theta_X + theta_X_tt
                else:
                    mask_tt = self.theta_masks[str(tup)]
                    theta_M_tt = mask_tt[None] * X_tt

                    if tup!=():
                        axes_to_sum_tt = list(range(1,len(tup)+1))
                        theta_M_tt = torch.sum(theta_M_tt, dim=axes_to_sum_tt)
                        
                    theta_MM_tt = torch.zeros_like(theta_M_tt)
                    theta_MM_tt[theta_M_tt==1] = -float('inf')
                    
                    theta_X = theta_X + (1-theta_M_tt)*theta_X_tt + theta_MM_tt
                pass
        return theta_X
    

    def recenter_theta_naught_for_split_model(self):
        #print("RECENTERING SPLIT MODEL")
        curr_device = self.theta_parameters["()"].device
        zbs = 10*1000
    
        final_hemi_LSE_list = []
        for ss,split in enumerate(self.maximal_splitting):
            split_tup = tuple(split)
            LSEs_ss = []
            all_good_idx_tensor = self.all_good_idx_tensor_list[ss]
            full_hemi_cumprod = all_good_idx_tensor.shape[0]
            # print('full_hemi_cumprod for_custom_backwards',full_hemi_cumprod,'zbs',zbs,"split hemi")

            semiglobal_div_integers = []
            semiglobal_mod_integers = [] #self.I_ks
            cumprod=1
            split_dict = {}
            for ii,i in enumerate(split_tup):
                I_i = self.I_ks[i]
                semiglobal_div_integers.append( cumprod )
                semiglobal_mod_integers.append(   I_i   )
                cumprod *= I_i
                split_dict[i] = ii
                
            curr_bs = zbs
            for zb in range( int(np.ceil(full_hemi_cumprod/curr_bs)) ):
                if (zb+1)*zbs>full_hemi_cumprod:
                    curr_bs = full_hemi_cumprod % zbs
                
                idx = all_good_idx_tensor[zb*zbs:zb*zbs+curr_bs].to(curr_device)
                onedim_tensors = self.create_split_onedim_tensors_from_global_indices(idx,curr_bs,curr_device,ss)
                theta_Z = self.get_batch_from_split_onedim_tensors(onedim_tensors,curr_bs,curr_device,ss,split_dict)
                
                LSE = torch.logsumexp(theta_Z,dim=0)
                LSEs_ss.append(LSE.detach())
            final_hemi_LSE_list.append(  torch.logsumexp(  torch.stack(LSEs_ss),dim=0  )  )
        pass
        # print('final_hemi_LSE_list')
        for ss,split in enumerate(self.maximal_splitting):
            # print(ss,final_hemi_LSE_list[ss])
            self.split_theta_naughts[ss] -= final_hemi_LSE_list[ss] #renormalization of the sub parts
        # print()

        if True: #also update the theta naught global here (to be used in trusted global model)
            theta_naught_param = self.theta_parameters[str(())]
            theta_naught_param.data = torch.sum(torch.stack(tuple(self.split_theta_naughts)))





    def forward_log_batch(self, x):
        onedim_tensors = self.split_apart_x_batch(x)
        theta_X = self.get_batch_from_onedim_tensors(onedim_tensors,curr_bs=x.shape[0],curr_device=x.device)
        return (theta_X)
    

    #kind-of helper which does the logic for pushing through a batch (given the onedim tensor format)
    def get_batch_from_onedim_tensors(self,onedim_tensors,curr_bs,curr_device):
        theta_X = 0
        for tt,tup in enumerate(self.my_inters):
            theta_tt = self.theta_parameters[str(tup)]*self.const_parameters[str(tup)]
            # if not self.theta_masks is None:
            #     theta_tt = theta_tt +  self.theta_masks[str(tup)] #MASKING by neg inf (additively)
        
            cumprod=1
            X_tt = torch.ones( (curr_bs,cumprod),device=curr_device )
            for i in tup:
                X_tt = torch.einsum("ij,ik->ijk",X_tt, onedim_tensors[i])
                cumprod *= self.I_ks[i]
                X_tt = X_tt.reshape( (-1,cumprod) )
            X_tt = X_tt.reshape( tuple([-1]+[self.I_ks[i] for i in tup]) )

            theta_X_tt = theta_tt[None] * X_tt
            if tup!=(): #now this correction is in the same place everywhere.
                axes_to_sum_tt = list(range(1,len(tup)+1))
                theta_X_tt = torch.sum(theta_X_tt, dim=axes_to_sum_tt)


            if self.theta_masks is None:
                theta_X = theta_X + theta_X_tt
            else:
                mask_tt = self.theta_masks[str(tup)]
                theta_M_tt = mask_tt[None] * X_tt

                if tup!=():
                    axes_to_sum_tt = list(range(1,len(tup)+1))
                    theta_M_tt = torch.sum(theta_M_tt, dim=axes_to_sum_tt)
                    
                theta_MM_tt = torch.zeros_like(theta_M_tt)
                theta_MM_tt[theta_M_tt==1] = -float('inf')
                
                theta_X = theta_X + (1-theta_M_tt)*theta_X_tt + theta_MM_tt
            pass
        return theta_X
    
    def get_oneTup_batch_from_onedim_tensors(self,onedim_tensors,curr_bs,curr_device,my_tup):
        theta_X = 0
        ###for tt,tup in enumerate(self.my_inters):
        for tup in [my_tup]:
            theta_tt = self.theta_parameters[str(tup)]*self.const_parameters[str(tup)]
            # if not self.theta_masks is None:
            #     theta_tt = theta_tt +  self.theta_masks[str(tup)] #MASKING by neg inf (additively)
        
            cumprod=1
            X_tt = torch.ones( (curr_bs,cumprod),device=curr_device )
            for i in tup:
                X_tt = torch.einsum("ij,ik->ijk",X_tt, onedim_tensors[i])
                cumprod *= self.I_ks[i]
                X_tt = X_tt.reshape( (-1,cumprod) )
            X_tt = X_tt.reshape( tuple([-1]+[self.I_ks[i] for i in tup]) )

            theta_X_tt = theta_tt[None] * X_tt
            if tup!=(): #now this correction is in the same place everywhere.
                axes_to_sum_tt = list(range(1,len(tup)+1))
                theta_X_tt = torch.sum(theta_X_tt, dim=axes_to_sum_tt)


            if self.theta_masks is None:
                theta_X = theta_X + theta_X_tt
            else:
                mask_tt = self.theta_masks[str(tup)]
                theta_M_tt = mask_tt[None] * X_tt

                if tup!=():
                    axes_to_sum_tt = list(range(1,len(tup)+1))
                    theta_M_tt = torch.sum(theta_M_tt, dim=axes_to_sum_tt)
                    
                # print(tup)
                theta_MM_tt = torch.zeros_like(theta_M_tt)
                theta_MM_tt[theta_M_tt==1] = -float('inf')
                
                theta_X = theta_X + (1-theta_M_tt)*theta_X_tt + theta_MM_tt
            pass
        return theta_X
    
    def get_batch_from_fulldim_tensor(self,x,curr_bs,curr_device):
        theta_X = 0
        for tt,tup in enumerate(self.my_inters):
            theta_tt = self.theta_parameters[str(tup)]*self.const_parameters[str(tup)]
            # if not self.theta_masks is None:
            #     theta_tt = theta_tt +  self.theta_masks[str(tup)] #MASKING by neg inf (additively)
        
            # cumprod=1
            # X_tt = torch.ones( (curr_bs,cumprod),device=curr_device )
            # for i in tup:
            #     X_tt = torch.einsum("ij,ik->ijk",X_tt, onedim_tensors[i])
            #     cumprod *= self.I_ks[i]
            #     X_tt = X_tt.reshape( (-1,cumprod) )
            # X_tt = X_tt.reshape( tuple([-1]+[self.I_ks[i] for i in tup]) )
            X_tt = self.fast_eta_reshaped_batching(x,tup,curr_device)

            theta_X_tt = theta_tt[None] * X_tt
            if tup!=(): #now this correction is in the same place everywhere.
                axes_to_sum_tt = list(range(1,len(tup)+1))
                theta_X_tt = torch.sum(theta_X_tt, dim=axes_to_sum_tt)


            if self.theta_masks is None:
                theta_X = theta_X + theta_X_tt
            else:
                mask_tt = self.theta_masks[str(tup)]
                theta_M_tt = mask_tt[None] * X_tt

                if tup!=():
                    axes_to_sum_tt = list(range(1,len(tup)+1))
                    theta_M_tt = torch.sum(theta_M_tt, dim=axes_to_sum_tt)
                    
                # print(tup)
                theta_MM_tt = torch.zeros_like(theta_M_tt)
                theta_MM_tt[theta_M_tt==1] = -float('inf')
                
                theta_X = theta_X + (1-theta_M_tt)*theta_X_tt + theta_MM_tt
            pass
        return theta_X
    
    
    def split_validate_against_peaks(self, split_onedim_tensors, curr_split, split_dict):
        curr_bs = split_onedim_tensors[0].shape[0]
        curr_device = split_onedim_tensors[0].device
        bad_hits = torch.zeros(curr_bs, dtype=bool, device=curr_device)

        split_tup = tuple(self.maximal_splitting[curr_split])

        for tt,tup in enumerate(self.my_inters):

            if tup!=() and tup[0] in split_tup:
                bad_tensors_tt = self.bad_tens_dict[tup]
                if len(bad_tensors_tt)>0:
                    conglommed_tensor = torch.cat([split_onedim_tensors[split_dict[i]] for i in tup],dim=-1) 

                    compare_tens = (conglommed_tensor[:,None,:]==bad_tensors_tt[None,:,:])
                    compare_tens = torch.all(compare_tens,dim=-1) #exact match for all the tuples dimensions
                    compare_tens = torch.any(compare_tens,dim=1)  #only matches one bad tensor from the list
                    


                    bad_hits = bad_hits + compare_tens
        return bad_hits
            
    def group_splittings_according_to_tup(self,splitting,tup):
        tup_split_inds = []
        for ii in range(len(tup)):
            i=tup[ii]
            s1=-1
            for s,split in enumerate(splitting):
                if i in split:
                    s1=s

            if s1 not in tup_split_inds:
                tup_split_inds.append(s1)
        return tup_split_inds

    def precompute_maximal_splitting(self):
        #COMPUTE THE MAXIMAL SPLITTING
        D=len(self.I_ks)
        maximal_splitting = [[i] for i in range(D)]
        print('maximal_splitting')
        print(maximal_splitting)
        for tt,tup in enumerate(self.my_inters):
            if len(tup)>1:
                tup_split_inds = self.group_splittings_according_to_tup(maximal_splitting,tup)
                if len(tup_split_inds)>1:
                    tup_split_inds=sorted(tup_split_inds,reverse=True)
                    popped_splits = [maximal_splitting.pop(s1) for s1 in tup_split_inds]
                    new_split = list(chain(*popped_splits))
                    maximal_splitting.append( new_split )

        maximal_splitting = [sorted(split) for split in maximal_splitting]
        print('maximal_splitting')
        print(maximal_splitting)
        self.maximal_splitting = maximal_splitting

    def initialize_unif_valid_hemi_indices(self):
        curr_device = self.theta_parameters["()"].device 
        zbs = 10*1000
        self.split_theta_naughts = nn.ParameterList()
        for ss, split in enumerate(self.maximal_splitting):
            self.split_theta_naughts.append( nn.Parameter( torch.zeros( () ).to(curr_device) ) )
            
        self.all_good_idx_tensor_list = []
        for s,split in enumerate(self.maximal_splitting):
            split_tup = tuple(split)
            split_dict = {}
            for ii,i in enumerate(split_tup):
                split_dict[i] = ii

            curr_bs = zbs
            full_cumprod = self.get_split_cumprod(split)
            all_good_idxes = []
            for zb in range( int(np.ceil(full_cumprod/curr_bs)) ):
                if (zb+1)*zbs>full_cumprod:
                    curr_bs = full_cumprod % zbs
                
                idx = torch.arange(curr_bs, device=curr_device) + zb*zbs
                split_onedim_tensors = self.create_split_onedim_tensors_from_semiglobal_indices(idx,curr_bs,curr_device,s)                 
                if True:
                    idx_global = [Z_k for ii,Z_k in enumerate(split_onedim_tensors)]
                    idx_global = [(Z_k.cpu().long()) for ii,Z_k in enumerate(split_onedim_tensors)]
                    idx_global = [torch.matmul(Z_k.cpu().long(),torch.arange(self.I_ks[split_tup[ii]]).cpu()) for ii,Z_k in enumerate(split_onedim_tensors)]
                    idx_global = [self.div_I_ks[split_tup[ii]] for ii,Z_k in enumerate(split_onedim_tensors)]
                    idx_global = [torch.matmul(Z_k.cpu().long(),torch.arange(self.I_ks[split_tup[ii]]).cpu().long())*self.div_I_ks[split_tup[ii]] for ii,Z_k in enumerate(split_onedim_tensors)]
                    idx_global = [torch.matmul(Z_k.cpu().long(),torch.arange(self.I_ks[split_tup[ii]]).cpu())*self.div_I_ks[split_tup[ii]] for ii,Z_k in enumerate(split_onedim_tensors)]
                    idx_global = torch.sum(torch.stack(idx_global,dim=0),dim=0)#.long()
                    good_idx = idx_global
                all_good_idxes.append(good_idx.cpu())
            all_good_idx_tensor = torch.cat(all_good_idxes,dim=-1)
            self.all_good_idx_tensor_list.append( all_good_idx_tensor )



    def precompute_all_valid_hemi_indices(self):
        curr_device = self.theta_parameters["()"].device 
        zbs = 10*1000

        self.split_theta_naughts = nn.ParameterList()
        for ss, split in enumerate(self.maximal_splitting):
            self.split_theta_naughts.append( nn.Parameter( torch.zeros( () ).to(curr_device) ) )
            
        #COMPUTE THE GOOD HEMI INDICES
        self.all_good_idx_tensor_list = []
        for s,split in enumerate(self.maximal_splitting):
            split_tup = tuple(split)
            split_dict = {}
            for ii,i in enumerate(split_tup):
                split_dict[i] = ii

            curr_bs = zbs
            full_cumprod = self.get_split_cumprod(split)
            all_good_idxes = []
            for zb in range( int(np.ceil(full_cumprod/curr_bs)) ):
                if (zb+1)*zbs>full_cumprod:
                    curr_bs = full_cumprod % zbs
                
                idx = torch.arange(curr_bs, device=curr_device) + zb*zbs
                split_onedim_tensors = self.create_split_onedim_tensors_from_semiglobal_indices(idx,curr_bs,curr_device,s) 
                if self.theta_masks is not None:
                    bad_hits = self.split_validate_against_peaks(split_onedim_tensors,s,split_dict)
                

                if True:
                    idx_global = [Z_k for ii,Z_k in enumerate(split_onedim_tensors)]
                    idx_global = [(Z_k.cpu().long()) for ii,Z_k in enumerate(split_onedim_tensors)]
                    idx_global = [torch.matmul(Z_k.cpu().long(),torch.arange(self.I_ks[split_tup[ii]]).cpu()) for ii,Z_k in enumerate(split_onedim_tensors)]
                    idx_global = [self.div_I_ks[split_tup[ii]] for ii,Z_k in enumerate(split_onedim_tensors)]
                    idx_global = [torch.matmul(Z_k.cpu().long(),torch.arange(self.I_ks[split_tup[ii]]).cpu().long())*self.div_I_ks[split_tup[ii]] for ii,Z_k in enumerate(split_onedim_tensors)]
                    idx_global = [torch.matmul(Z_k.cpu().long(),torch.arange(self.I_ks[split_tup[ii]]).cpu())*self.div_I_ks[split_tup[ii]] for ii,Z_k in enumerate(split_onedim_tensors)]
                    idx_global = torch.sum(torch.stack(idx_global,dim=0),dim=0)#.long()
                    if self.theta_masks is not None:
                        good_idx = idx_global[~bad_hits.cpu()]
                    else:
                        print('maybe things are bad')
                        good_idx = idx_global
                all_good_idxes.append(good_idx.cpu())
            all_good_idx_tensor = torch.cat(all_good_idxes,dim=-1)
            print(split,'\t',all_good_idx_tensor.shape)
            self.all_good_idx_tensor_list.append( all_good_idx_tensor )
        pass

    def precompute_all_split_hemi_taus(self):
        curr_device = self.theta_parameters["()"].device
        zbs = 10*1000 #magic number

        self.my_hemi_uniform_dict = {}
        for tt,tup in enumerate(self.my_inters):
            self.my_hemi_uniform_dict[tup] = torch.zeros_like(  self.theta_parameters[str(tup)], device=curr_device  )


        for s,split in enumerate(self.maximal_splitting):
            split_tup = tuple(split)
            split_dict = {}
            for ii,i in enumerate(split_tup):
                split_dict[i] = ii
            pass

            curr_bs = zbs
            all_good_idx_tensor = self.all_good_idx_tensor_list[s]
            full_hemi_cumprod = all_good_idx_tensor.shape[0]

            
            for zb in range( int(np.ceil(full_hemi_cumprod/curr_bs)) ):
                if (zb+1)*zbs>full_hemi_cumprod:
                    curr_bs = full_hemi_cumprod % zbs
                idx = all_good_idx_tensor[zb*zbs:zb*zbs+curr_bs].to(curr_device)
                split_onedim_tensors = self.create_split_onedim_tensors_from_global_indices(idx,curr_bs,curr_device,s)

                for tt,tup in enumerate(self.my_inters):
                    if tup!=() and tup[0] in split_tup: #forgot this change as well
                        cumprod=1
                        Z_tt = torch.ones( (curr_bs,cumprod),device=curr_device )
                        for i in tup:
                            ii = split_dict[i]
                            Z_tt = torch.einsum("ij,ik->ijk",Z_tt, split_onedim_tensors[ii])
                            cumprod *= self.I_ks[i]
                            Z_tt = Z_tt.reshape( (-1,cumprod) )
                        Z_tt = Z_tt.reshape( tuple([-1]+[self.I_ks[i] for i in tup]) )
                        
                        self.my_hemi_uniform_dict[tup] += torch.sum(Z_tt,dim=0)

            pass
        for tt,tup in enumerate(self.my_inters):
            if tup!=():
                self.my_hemi_uniform_dict[tup] = self.my_hemi_uniform_dict[tup] / torch.sum(self.my_hemi_uniform_dict[tup])
            else:
                self.my_hemi_uniform_dict[tup] = self.my_hemi_uniform_dict[tup] + 1.0
        pass




    def vibe_check(self,w):
        curr_bs = w.shape[0]
        curr_device = w.device
        onedim_tensors = self.split_apart_x_batch(w.clone())
        print('energy check outside vibe',torch.min(self.get_batch_from_onedim_tensors(onedim_tensors,curr_bs,curr_device)))


    def one_round_of_Gibbs_updates(self, w):
        with torch.no_grad():
            curr_bs = w.shape[0]
            curr_device = w.device
            w_new = w.clone()
            onedim_tensors = self.split_apart_x_batch(w_new.clone())

            def safe_softmax(logits):
                logits_max, _ = torch.max(logits,dim=-1,keepdim=True)
                logits_exp = torch.exp(logits-logits_max)
                logits_sum = torch.sum(logits_exp,dim=-1,keepdim=True)
                logits_SM = logits_exp / logits_sum
                

                if torch.any(torch.isnan(logits-logits_max)):
                    print("SCREAM UNSAFE")
                if torch.any(torch.isnan(logits_SM)):
                    print("SCREAM UNSAFE 2")
                return logits_SM

            D = len(self.I_ks)
            for i in range(D):
                conditional_energies_i = []
                I_i = self.I_ks[i]
                
                for event_i in range(I_i):
                    new_W_i = torch.zeros_like(onedim_tensors[i])
                    # print('new_W_i',new_W_i.shape,'event_i',event_i)
                    new_W_i[:,event_i] = 1
                    onedim_tensors[i] = new_W_i
                    energy_ei = self.get_batch_from_onedim_tensors(onedim_tensors,curr_bs,curr_device) 
                    
                    conditional_energies_i.append(energy_ei)
                conditional_energies_i = torch.stack(conditional_energies_i,dim=-1)

            
                u_i = torch.ones_like(conditional_energies_i)
                if I_i>1:
                    u_i = safe_softmax(conditional_energies_i)
                Z_i = torch.zeros_like( u_i, device=curr_device ).scatter_(1, torch.multinomial(u_i,1), 1.)
                onedim_tensors[i] = Z_i

            w_new = torch.cat( onedim_tensors, dim=-1 )
            return w_new  

    def one_round_of_block_Gibbs_updates(self, w):
        with torch.no_grad():
            for tt,tup in enumerate(self.my_inters):
                if tup!=():
                    w = self.one_step_of_block_Gibbs_updates(w, tup)
            return w
        
        
    def one_partialRound_of_block_Gibbs_updates(self, w, amount=None, perc=None,   temp=None, old_model=None):
        amount=amount
        TT = len(self.my_inters)-1

        if perc is not None:
            amount = math.ceil(perc*TT)
        if amount is None:
            amount = 0 
        if amount>TT:
            amount=TT

        with torch.no_grad():
            perm = np.random.permutation(TT)[:amount]+1 #offset by null tuple
            for tt in perm:
                tup = self.my_inters[tt]
                if tup!=():
                    w = self.one_step_of_block_Gibbs_updates(w, tup, temp=temp,old_model=old_model)
            return w
        
    def random_block_Gibbs_updates(self, w, amount=10):
        with torch.no_grad():
            for _ in range(amount):
                tt = np.random.randint(len(self.my_inters))
                tup = self.my_inters[tt]
                if tup!=():
                    w = self.one_step_of_block_Gibbs_updates(w, tup)
            return w
        
    def one_partialNewRound_of_block_Gibbs_updates(self, w):
        with torch.no_grad():
            for tt,tup in enumerate(self.new_tups):
                if tup!=():
                    w = self.one_step_of_block_Gibbs_updates(w, tup)
            return w

    def one_step_of_block_Gibbs_updates(self, w, my_tup, temp=None, old_model=None):
        with torch.no_grad():
            curr_bs = w.shape[0]
            curr_device = w.device
            w_new = w.clone()
            onedim_tensors = self.split_apart_x_batch(w_new.clone())

            def safe_softmax(logits):
                logits_max, _ = torch.max(logits,dim=-1,keepdim=True)
                logits_exp = torch.exp(logits-logits_max)
                logits_sum = torch.sum(logits_exp,dim=-1,keepdim=True)
                logits_SM = logits_exp / logits_sum
                return logits_SM



            K=len(my_tup)
            event_space = torch.zeros(tuple([K]+[1]*K),dtype=int)
            for ii,i in enumerate(my_tup):
                I_i = self.I_ks[i]
                exp_tup = tuple([None]+[None]*ii+[slice(None)]+[None]*(K-ii-1))
                event_space_i = torch.zeros(tuple([K]+[1]*ii+[I_i]+[1]*(K-ii-1)),dtype=int) #should be (K x 1x1x ... x I_k x ... x1x1)
                event_space_i[ii] = torch.arange(I_i)[exp_tup]  #should be the same size
                event_space = event_space + event_space_i
            event_space = event_space.reshape(K,-1)
            event_space = torch.transpose(event_space,0,1)

            conditional_energies_tt = []
            conditional_energies_tt = [0.0] * len(event_space)
            for tt,tup in enumerate(self.my_inters):
                int_tup = tuple([i for i in my_tup if i in tup])
                energy_cache = {}
                for ee,event_tt in enumerate(event_space):
                    event_to_cache = tuple([event_tt[ii] for ii,i in enumerate(my_tup) if i in int_tup])

                    if event_to_cache not in energy_cache:
                        for ii,i in enumerate(my_tup):
                            if i in int_tup: #only update this when necessary
                                event_i = event_tt[ii]
                                new_W_i = torch.zeros_like(onedim_tensors[i])
                                new_W_i[:,event_i] = 1
                                onedim_tensors[i] = new_W_i
                        if True: 
                            #energy for this tuple (tt), this feature (ii), this event (ii_ee) 
                            energy_tt_ii_ee = self.get_oneTup_batch_from_onedim_tensors(onedim_tensors,curr_bs,curr_device,tup)
                        energy_cache[event_to_cache] = energy_tt_ii_ee
                    else:
                        energy_tt_ii_ee = energy_cache[event_to_cache]
                    conditional_energies_tt[ee] += energy_tt_ii_ee

            conditional_energies_tt = torch.stack(conditional_energies_tt,dim=-1)
            if temp is not None: #dampen the theta (mixture with the 'hot' uniform distribution); note: beta is sometimes called the 'coldness function'
                if old_model is None:
                    conditional_energies_tt = temp*conditional_energies_tt
                else:
                    conditional_energies_tt_old = None
                    ####    ####
                    conditional_energies_tt_old = [0.0] * len(event_space)
                    for tt,tup in enumerate(old_model.my_inters):
                        int_tup = tuple([i for i in my_tup if i in tup])
                        energy_cache = {}
                        for ee,event_tt in enumerate(event_space):
                            event_to_cache = tuple([event_tt[ii] for ii,i in enumerate(my_tup) if i in int_tup])
                            if event_to_cache not in energy_cache:
                                for ii,i in enumerate(my_tup):
                                    if i in int_tup: #only update this when necessary
                                        event_i = event_tt[ii]; new_W_i = torch.zeros_like(onedim_tensors[i]); new_W_i[:,event_i] = 1; onedim_tensors[i] = new_W_i;
                                energy_tt_ii_ee = old_model.get_oneTup_batch_from_onedim_tensors(onedim_tensors,curr_bs,curr_device,tup)
                                energy_cache[event_to_cache] = energy_tt_ii_ee
                            else:
                                energy_tt_ii_ee = energy_cache[event_to_cache]
                            conditional_energies_tt_old[ee] += energy_tt_ii_ee
                    conditional_energies_tt_old = torch.stack(conditional_energies_tt_old,dim=-1)
                    ####    ####
                    conditional_energies_tt = conditional_energies_tt_old + temp*(conditional_energies_tt-conditional_energies_tt_old)


            u_tt = safe_softmax(conditional_energies_tt)
            to_discard = torch.any(torch.isnan(u_tt),dim=-1)
            
            Z_tt = torch.zeros_like( u_tt, device=curr_device )
            fake = torch.ones(u_tt.shape[1], device=curr_device ); fake=fake/torch.sum(fake);
            u_tt[to_discard] = fake
            Z_tt = torch.zeros_like( u_tt, device=curr_device ).scatter_(1, torch.multinomial(u_tt,1), 1.)
            Z_tt = Z_tt.reshape( tuple([-1]+[self.I_ks[i] for i in my_tup]) )
            

            for ii,i in enumerate(my_tup):
                axes_to_sum = list(range(1,K+1))
                axes_to_sum.remove(ii+1)
                if len(axes_to_sum)>0:
                    onedim_tensors[i] = torch.sum( Z_tt, dim=tuple(axes_to_sum) )
                else:
                    onedim_tensors[i] = Z_tt
                    

            if torch.sum(to_discard.int())>0:
                OG_size = onedim_tensors[0].shape[0]
                print('(need to discard',int(torch.sum(to_discard.int())),'/',OG_size,'W samples)')

                D=len(onedim_tensors)
                for i in range(D):
                    W_i = onedim_tensors[i][~to_discard]
                    assert W_i.shape[0]>0, "hey, we are out of samples"
                    necessary_repeats = OG_size//W_i.shape[0] + 1
                    if necessary_repeats>2:
                        W_i = W_i.repeat(necessary_repeats,1)
                        W_i = W_i[:OG_size]
                    else:
                        assert necessary_repeats==2
                        W_i = torch.cat([W_i,W_i[:(OG_size-W_i.shape[0])]],dim=0)
                    pass
                    onedim_tensors[i] = W_i 

            w_new = torch.cat( onedim_tensors, dim=-1 )
            return w_new  
    

   
    
    #HELPER FN
    def split_apart_x_batch(self,x):
        onedim_tensors = []
        D=len(self.I_ks)
        cum_n=0
        for i in range(D):
            I_i = self.I_ks[i]
            X_i = x[:,cum_n:cum_n+I_i]
            cum_n += I_i  
            onedim_tensors.append( X_i )
        return onedim_tensors
    

    #HELPER FN
    def get_cumprod(self):
        cumprod=1
        D=len(self.I_ks)
        for i in range(D):
            I_i = self.I_ks[i]
            cumprod *= I_i
        return cumprod
    
    def get_split_cumprod(self,split):
        cumprod=1
        for i in split:
            I_i = self.I_ks[i]
            cumprod *= I_i
        return cumprod

    def get_hemi_cumprod(self): 
        cumprod=1
        for s,split in enumerate(self.maximal_splitting):
            all_good_idx_tensor = self.all_good_idx_tensor_list[s]
            cumprod *= all_good_idx_tensor.shape[0]
        return cumprod
    
    def generate_uniformReplacing_z_batch(self,full_zbs,return_split_apart=False):
        curr_device = self.theta_parameters["()"].device

        onedim_tensors = []
        D=len(self.I_ks)
        for i in range(D):
            u_i = (torch.ones( (full_zbs,self.I_ks[i]) )/self.I_ks[i]).to(curr_device)
            Z_i = torch.zeros_like( u_i, device=curr_device ).scatter_(1, torch.multinomial(u_i,1), 1.)
            onedim_tensors.append( Z_i )

        if return_split_apart:
            return onedim_tensors
        else:
            Z_D = torch.cat( onedim_tensors, dim=-1 )
            return Z_D
    

    def generate_hemiUniformReplacing_z_batch(self,full_zbs,return_split_apart=False): 
        curr_device = self.theta_parameters["()"].device

        total_valid = 0
        one_dim_tensors_list = []
        D=len(self.I_ks)
        while (total_valid < full_zbs):

            onedim_tensors_try = []
            for i in range(D):
                u_i = (torch.ones( (full_zbs,self.I_ks[i]) )/self.I_ks[i]).to(curr_device)
                Z_i = torch.zeros_like( u_i, device=curr_device ).scatter_(1, torch.multinomial(u_i,1), 1.)
                onedim_tensors_try.append( Z_i )

            bad_hits_all = torch.zeros(full_zbs,dtype=bool,device=curr_device)
            for s,split in enumerate(self.maximal_splitting):
                split_tup = tuple(split)
                split_dict = {}
                for ii,i in enumerate(split_tup):
                    split_dict[i] = ii

                split_onedim_tensors = [onedim_tensors_try[i] for i in split]
                bad_hits = self.split_validate_against_peaks(split_onedim_tensors,s,split_dict)
                bad_hits_all = bad_hits_all + bad_hits #think this works

            onedim_tensors_try = [Z_i[~bad_hits] for Z_i in onedim_tensors_try]
            total_valid+=onedim_tensors_try[0].shape[0]
            print('total_valid',total_valid)
            one_dim_tensors_list.append(onedim_tensors_try)

        onedim_tensors = one_dim_tensors_list 
        onedim_tensors = [torch.cat([thing[i] for thing in one_dim_tensors_list]) for i in range(D)]
        print(onedim_tensors[0].shape)
        onedim_tensors = [thing[:full_zbs] for thing in onedim_tensors]
        print(onedim_tensors[0].shape)

        if return_split_apart:
            return onedim_tensors
        else:
            Z_D = torch.cat( onedim_tensors, dim=-1 )
            return Z_D
        

    def full_flattened_log_forward(self):

        z = self.generate_completed_z_batch(False)
        onedim_tensors = self.split_apart_x_batch(z)
        curr_device = self.theta_parameters["()"].device
        curr_bs = onedim_tensors[0].shape[0]

        if True: #might need batching to fit on memory
            theta_X = self.get_batch_from_onedim_tensors(onedim_tensors,curr_bs,curr_device)
                    
        LSE = torch.logsumexp(theta_X,dim=0)
        return z, (theta_X - LSE)
    
    def semifull_flattened_log_forward(self,zbs=None):
        if zbs is None:
            zbs = 10*1000
        curr_device = self.theta_parameters["()"].device
        curr_bs = zbs
        onedim_tensors = self.generate_uniformReplacing_z_batch(zbs,curr_device)

        if True: #might need batching to fit on memory
            theta_X = self.get_batch_from_onedim_tensors(onedim_tensors,curr_bs,curr_device)
                    
        LSE = torch.logsumexp(theta_X,dim=0)
        return onedim_tensors, theta_X, LSE
    
    def batch_semifull_flattened_log_forward_with_backwards(self,full_zbs=None,zbs=None):
        if full_zbs is None:
            full_zbs = 10*1000
        if zbs is None:
            zbs = 10*1000
        curr_device = self.theta_parameters["()"].device
        curr_zbs = zbs
        
        full_cumprod = self.get_cumprod()
        D=len(self.I_ks)

        if True: #might need batching to fit on memory
            onedim_tensors = self.generate_uniformReplacing_z_batch(curr_zbs,curr_device)
            theta_X = self.get_batch_from_onedim_tensors(onedim_tensors,curr_zbs,curr_device)
        LSE = torch.logsumexp(theta_X,dim=0)



        LSEs = []
        for zb in range( int(np.ceil(full_zbs/zbs)) ):
            if (zb+1)*zbs>full_zbs:
                curr_zbs = full_zbs % zbs
            
            onedim_tensors = self.generate_uniformReplacing_z_batch(curr_zbs,curr_device)
            theta_X = self.get_batch_from_onedim_tensors(onedim_tensors,curr_zbs,curr_device)
            LSE_hat = torch.logsumexp(theta_X,dim=0)
            LSE_hat = LSE_hat + (np.log(full_cumprod)-np.log(full_zbs))
            
            if True:
                SE = torch.exp(LSE_hat)
                SE.backward()
                
            if True: #hopefully this will release everything
                del onedim_tensors
                del theta_X
            LSEs.append(LSE.detach())
        return None




    def forward(self):        
        return self.full_expanded_forward()
    def full_expanded_forward(self):
        return None


    def full_trusted_forward(self):
        curr_device = self.theta_parameters["()"].device
        new_logits = torch.zeros(*self.I_ks).to(curr_device)
        verbose=False

        for tt,tup in enumerate(self.my_inters):
            theta_tt = self.theta_parameters[str(tup)]*self.const_parameters[str(tup)]
            if verbose:
                print('tt',tt,'tup',tup)
                print('theta_tt',theta_tt.shape)

            extended_tup = []
            for d in range(len(self.I_ks)):
                if d in tup:
                    extended_tup.append(slice(None))
                else:
                    extended_tup.append(None)
            extended_tup=tuple(extended_tup)

            if verbose:
                print(tup,'\t',theta_tt[extended_tup].shape)
            new_logits += theta_tt[extended_tup]
            
            if self.theta_masks is not None:
                mask_tt = self.theta_masks[str(tup)]
                theta_MM_tt = torch.zeros_like(new_logits)
                theta_MM_tt += mask_tt[extended_tup]
                new_logits[theta_MM_tt==1] = -float('inf')            
            
        return new_logits



    def cond_pred_forward(self, x, class_idx=None):
        with torch.no_grad():
            curr_bs = x.shape[0]
            curr_device = x.device
            x_new = x.clone()
            onedim_tensors = self.split_apart_x_batch(x_new.clone())

            def safe_softmax(logits):
                small_mask = (logits==-float('inf'))
                logits_max, _ = torch.max(logits,dim=-1,keepdim=True)
                logits_exp = torch.exp(logits-logits_max)
                logits_sum = torch.sum(logits_exp,dim=-1,keepdim=True)
                logits_SM = logits_exp / logits_sum
                if True: #will this help??
                    logits_SM[small_mask] = 0.0
                return logits_SM


            my_tup = (class_idx,)
            K=len(my_tup)
            
            event_space = torch.zeros(tuple([K]+[1]*K),dtype=int)
            for ii,i in enumerate(my_tup):
                I_i = self.I_ks[i]
                exp_tup = tuple([None]+[None]*ii+[slice(None)]+[None]*(K-ii-1))
                event_space_i = torch.zeros(tuple([K]+[1]*ii+[I_i]+[1]*(K-ii-1)),dtype=int) #should be (K x 1x1x ... x I_k x ... x1x1)
                event_space_i[ii] = torch.arange(I_i)[exp_tup]  #should be the same size
                event_space = event_space + event_space_i
            event_space = event_space.reshape(K,-1)
            event_space = torch.transpose(event_space,0,1)

            conditional_energies_tt = [0.0] * len(event_space)
            for tt,tup in enumerate(self.my_inters):
                int_tup = tuple([i for i in my_tup if i in tup]) 
                energy_cache = {}
                for ee,event_tt in enumerate(event_space):
                    event_to_cache = tuple([event_tt[ii] for ii,i in enumerate(my_tup) if i in int_tup])

                    if event_to_cache not in energy_cache:
                        for ii,i in enumerate(my_tup):
                            if i in int_tup: #only update this when necessary
                                event_i = event_tt[ii]
                                new_W_i = torch.zeros_like(onedim_tensors[i])
                                new_W_i[:,event_i] = 1
                                onedim_tensors[i] = new_W_i
                        if True: 
                            #energy for this tuple (tt), this feature (ii), this event (ii_ee) 
                            energy_tt_ii_ee = self.get_oneTup_batch_from_onedim_tensors(onedim_tensors,curr_bs,curr_device,tup)
                        energy_cache[event_to_cache] = energy_tt_ii_ee
                    else:
                        energy_tt_ii_ee = energy_cache[event_to_cache]
                    conditional_energies_tt[ee] += energy_tt_ii_ee

            conditional_energies_tt = torch.stack(conditional_energies_tt,dim=-1)

            u_tt = safe_softmax(conditional_energies_tt)
            to_discard = torch.any(torch.isnan(u_tt),dim=-1); fake = torch.ones(u_tt.shape[1], device=curr_device ); fake=fake/torch.sum(fake);
            u_tt[to_discard] = fake
            return (u_tt)





    def get_group_lasso_norms(self):
        theta_two_norms = {}
        for tt,tup in enumerate(self.my_inters):
            theta_two_norms[tup] = torch.norm(self.theta_parameters[str(tup)], p=2)
        return theta_two_norms

    def recenter_theta_parameters(self):
        maxlen = 0
        for tt,tup in enumerate(self.my_inters):
            if len(tup)>maxlen:
                maxlen=len(tup)

        verbose=True
        verbose=False
        for currlen in range(maxlen,0,-1):
            if verbose:
                print()
                print()
                print('currlen',currlen)
            for tt,tup in enumerate(self.my_inters):
                if len(tup)==currlen:
                    #need to marginalize

                    if verbose:
                        print()
                    theta_tt = self.theta_parameters[str(tup)]
                    purepower = list(powerset(tup))
                    residuals_dict = {}
                    residuals_dict[tup] = torch.zeros_like(theta_tt)
                    for purelen in range(currlen-1,-1,-1): 

                        for pp,puretup in enumerate(purepower):
                            if len(puretup)==purelen:
                                pass
                                if verbose:
                                    print('pp',pp,'puretup',puretup)

                                exttup = tuple( [(slice(None) if i in puretup else None) for i in tup] )
                                axes_to_sum = [ii for ii,i in enumerate(tup) if (i not in puretup)] 
                                if verbose:
                                    print(pp,tup,puretup,exttup)
                                    print(axes_to_sum)


                                if verbose:
                                    print('theta_tt',theta_tt.shape) 
                                theta_residuals_pp = torch.mean(theta_tt, axis=tuple(axes_to_sum)) 
                                if verbose:
                                    print('theta_residuals_pp',theta_residuals_pp.shape)
                                    print('theta_residuals_pp[exttup]',theta_residuals_pp[exttup].shape)
                                    print(theta_residuals_pp)


                                sign = 0.0
                                if (purelen-currlen)%2 == 1: #add everything one below downwards (as 'residual') but subtract from here (by subtracting 'residual')
                                    sign = 1.0
                                else:
                                    sign = -1.0

                                if verbose:
                                    print('ahh not implem-d for torch str(tup)')
                                    print('theta_parameters[puretup]',self.theta_parameters[puretup].shape)
                                with torch.no_grad():
                                    self.theta_parameters[str(puretup)] += sign*theta_residuals_pp
                                if verbose:
                                    print('residuals_dict[tup]',residuals_dict[tup].shape)
                                residuals_dict[tup]       += sign*theta_residuals_pp[exttup]
                                pass
                    
                            
                    ''' 
                    '''  

                    with torch.no_grad(): 
                        self.theta_parameters[str(tup)] -= residuals_dict[(tup)]           
                    '''
                    '''

        if verbose:
            print("theta are renormalized :-)")
        pass

    def rechop_theta_parameters(self):
        maxlen = 0
        for tt,tup in enumerate(self.my_inters):
            if len(tup)>maxlen:
                maxlen=len(tup)

        verbose=True
        verbose=False
        for currlen in range(maxlen,0,-1):
            if verbose:
                print()
                print()
                print('currlen',currlen)
            for tt,tup in enumerate(self.my_inters):
                if len(tup)==currlen:
                    #need to marginalize

                    if verbose:
                        print()
                    theta_tt = self.theta_parameters[str(tup)]
                    purepower = list(powerset(tup))
                    residuals_dict = {}
                    residuals_dict[tup] = torch.zeros_like(theta_tt)
                    for purelen in range(currlen-1,-1,-1): 

                        for pp,puretup in enumerate(purepower):
                            if len(puretup)==purelen:
                                pass
                                if verbose:
                                    print('pp',pp,'puretup',puretup)

                                exttup = tuple( [(slice(None) if i in puretup else None) for i in tup] )
                                axes_to_sum = [ii for ii,i in enumerate(tup) if (i not in puretup)] 
                                if verbose:
                                    print(pp,tup,puretup,exttup)
                                    print(axes_to_sum)


                                if verbose:
                                    print('theta_tt',theta_tt.shape) 
                                theta_residuals_pp = torch.mean(theta_tt, axis=tuple(axes_to_sum)) 
                                if verbose:
                                    print('theta_residuals_pp',theta_residuals_pp.shape)
                                    print('theta_residuals_pp[exttup]',theta_residuals_pp[exttup].shape)
                                    print(theta_residuals_pp)

                                
                                sign = 0.0
                                if (purelen-currlen)%2 == 1: #add everything one below downwards (as 'residual') but subtract from here (by subtracting 'residual')
                                    sign = 1.0
                                else:
                                    sign = -1.0

                                if verbose:
                                    print('ahh not implem-d for torch str(tup)')
                                    print('theta_parameters[puretup]',self.theta_parameters[puretup].shape)
                                if verbose:
                                    print('residuals_dict[tup]',residuals_dict[tup].shape)
                                residuals_dict[tup]       += sign*theta_residuals_pp[exttup]
                                pass

                            
                    ''' 
                    '''
                    with torch.no_grad(): 
                        self.theta_parameters[str(tup)] -= residuals_dict[(tup)]           
                    '''
                    '''

        if verbose:
            print("theta are renormalized :-)")
        pass

    def recenter_theta_parameters_semiUniform(self):
        maxlen = 0
        for tt,tup in enumerate(self.my_inters):
            if len(tup)>maxlen:
                maxlen=len(tup)

        verbose=True
        verbose=False
        for currlen in range(maxlen,0,-1):
            if verbose:
                print()
                print()
                print('currlen',currlen)
            for tt,tup in enumerate(self.my_inters):
                if len(tup)==currlen:
                    #need to marginalize

                    if verbose:
                        print()
                    theta_tt = self.theta_parameters[str(tup)]
                    mask_tt = self.theta_masks[str(tup)]

                    purepower = list(powerset(tup))
                    residuals_dict = {}
                    residuals_dict[tup] = torch.zeros_like(theta_tt)
                    for purelen in range(currlen-1,-1,-1): 

                        for pp,puretup in enumerate(purepower):
                            if len(puretup)==purelen:
                                pass
                                if verbose:
                                    print('pp',pp,'puretup',puretup)

                                exttup = tuple( [(slice(None) if i in puretup else None) for i in tup] )
                                axes_to_sum = [ii for ii,i in enumerate(tup) if (i not in puretup)] 
                                if verbose:
                                    print(pp,tup,puretup,exttup)
                                    print(axes_to_sum)


                                if verbose:
                                    print('theta_tt',theta_tt.shape) 

                                if True:
                                    with torch.no_grad():
                                        theta_residuals_pp = torch.sum(theta_tt, axis=tuple(axes_to_sum))
                                        theta_masks_pp = torch.sum( (1-mask_tt), axis=tuple(axes_to_sum))
                                        theta_masks_pp[theta_masks_pp==0] = 1.0 #eliminate divide by zero error
                                        theta_residuals_pp = theta_residuals_pp / theta_masks_pp
                                    if torch.any(torch.isnan(theta_residuals_pp)):
                                        print("SCREAMING OUT NAN")
                                        print(tt,tup)
                                        print(theta_tt)
                                        print(mask_tt)
                                        print(theta_residuals_pp)
                                
                                if verbose:
                                    print('theta_residuals_pp',theta_residuals_pp.shape)
                                    print('theta_residuals_pp[exttup]',theta_residuals_pp[exttup].shape)
                                    print(theta_residuals_pp)


                                sign = 0.0
                                if (purelen-currlen)%2 == 1: #add everything one below downwards (as 'residual') but subtract from here (by subtracting 'residual')
                                    sign = 1.0
                                else:
                                    sign = -1.0

                                if verbose:
                                    print('ahh not implem-d for torch str(tup)')
                                    print('theta_parameters[puretup]',self.theta_parameters[puretup].shape)
                                with torch.no_grad():
                                    mask_pp = self.theta_masks[str(puretup)]
                                    self.theta_parameters[str(puretup)] += sign*theta_residuals_pp*(1-mask_pp)
                                    if torch.any(torch.isnan(sign*theta_residuals_pp*mask_pp)):
                                        print("WAIT ITS OVER HERE")
                                        print(sign*theta_residuals_pp*mask_pp)
                                        print(theta_residuals_pp)
                                        print(mask_pp)
                                if verbose:
                                    print('residuals_dict[tup]',residuals_dict[tup].shape)
                                residuals_dict[tup]       += sign*theta_residuals_pp[exttup]*(1-mask_tt)
                                pass
                    
                            
                    ''' 
                    '''
                    with torch.no_grad(): 
                        self.theta_parameters[str(tup)] -= residuals_dict[(tup)]           
                    '''
                    '''

        if verbose:
            print("theta are renormalized :-)")
        pass



    def purify_theta_gradients(self):
        return self.purify_theta_gradients_localTauHemiUnif_butMaskZero_vv()

    def purify_theta_gradients_semiUnif_butMaskZero_vf(self):
        maxlen = 0
        for tt,tup in enumerate(self.my_inters):
            if len(tup)>maxlen:
                maxlen=len(tup)


        curr_device = self.theta_parameters["()"].device
            

        verbose=True
        verbose=False
        for currlen in range(maxlen,0,-1):
            if verbose:
                print()
                print()
                print('currlen',currlen)
            for tt,tup in enumerate(self.my_inters):
                if len(tup)==currlen:
                    theta_tt = self.theta_parameters[str(tup)]
                    if self.theta_masks is not None: 
                        mask_tt = self.theta_masks[str(tup)]
                    else:
                        mask_tt = torch.zeros_like(theta_tt)

                    complete_semimasked_U = 1.0 - mask_tt
                    complete_semimasked_U = complete_semimasked_U / torch.sum(complete_semimasked_U)

                    purepower = list(powerset(tup))
                    residuals_dict = {}
                    residuals_dict[tup] = torch.zeros_like(theta_tt)
                    for purelen in range(currlen-1,-1,-1): 
                        for pp,puretup in enumerate(purepower):
                            if len(puretup)==purelen:
                                exttup1 = tuple( [(slice(None) if i in puretup else None) for i in tup] )
                                exttup2 = tuple( [(slice(None) if i not in puretup else None) for i in tup] )
                                axes_to_sum = tuple( [ii for ii,i in enumerate(tup) if i in puretup] )

                                if len(axes_to_sum)>0:
                                    unflattened_U = torch.sum( complete_semimasked_U, dim=axes_to_sum )
                                else:
                                    unflattened_U = complete_semimasked_U #maybe care referencing?
                                unflattened_U = unflattened_U[exttup2]
                                theta_grad_pp = self.theta_parameters[str(puretup)].grad[exttup1]
                                theta_grad_pp = self.theta_parameters[str(puretup)].grad[exttup1] * unflattened_U #pure approach
                                

                                sign = 0.0
                                if (purelen-currlen)%2 == 1:
                                    sign = 1.0
                                else:
                                    sign = -1.0

                                with torch.no_grad():
                                    self.theta_parameters[str(tup)].grad -= sign*theta_grad_pp


                    with torch.no_grad(): #zeroing out anything that is masked to be -inf
                        self.theta_parameters[str(tup)].grad = self.theta_parameters[str(tup)].grad * (1-mask_tt)  
                    '''
                    '''
        if verbose:
            print("theta grads are renormalized (vf) :-)")
        pass
    
    def purify_theta_gradients_localTauHemiUnif_butMaskZero_vv(self):
        maxlen = 0
        for tt,tup in enumerate(self.my_inters):
            if len(tup)>maxlen:
                maxlen=len(tup)
            

        verbose=True
        verbose=False
        for currlen in range(maxlen,0,-1):
            if verbose:
                print()
                print()
                print('currlen',currlen)
            for tt,tup in enumerate(self.my_inters):
                if len(tup)==currlen:
                    theta_tt = self.theta_parameters[str(tup)]
                    complete_hemimasked_U = self.my_hemi_uniform_dict[tup]

                    TURNING_ON_LOCAL_TAUS = False
                    if TURNING_ON_LOCAL_TAUS:
                        complete_current_eta = self.current_eta_dict[tup] 
                        complete_hemimasked_U = complete_current_eta

                    purepower = list(powerset(tup))
                    residuals_dict = {}
                    residuals_dict[tup] = torch.zeros_like(theta_tt)
                    for purelen in range(currlen-1,-1,-1): 
                        for pp,puretup in enumerate(purepower): #order doesnt actually matter for this one  (unlike the outside loop)
                            if len(puretup)==purelen:
                                pass

                                exttup1 = tuple( [(slice(None) if i in puretup else None) for i in tup] )
                                exttup2 = tuple( [(slice(None) if i not in puretup else None) for i in tup] )
                                axes_to_sum = tuple( [ii for ii,i in enumerate(tup) if i in puretup] )

                                if len(axes_to_sum)>0:
                                    unflattened_U = torch.sum( complete_hemimasked_U, dim=axes_to_sum )
                                else:
                                    unflattened_U = complete_hemimasked_U #maybe care referencing?
                                unflattened_U = unflattened_U[exttup2]
                                theta_grad_pp = self.theta_parameters[str(puretup)].grad[exttup1]
                                theta_grad_pp = self.theta_parameters[str(puretup)].grad[exttup1] * unflattened_U #pure approach
                                

                                sign = 0.0 #copied from theta_centering, but is basically the same
                                if (purelen-currlen)%2 == 1: #add everything one below downwards (as 'residual') but subtract from here (by subtracting 'residual')
                                    sign = 1.0
                                else:
                                    sign = -1.0

                                with torch.no_grad():
                                    self.theta_parameters[str(tup)].grad -= sign*theta_grad_pp
                    '''
                    '''

        if verbose:
            print("theta grads are renormalized (vv) :-)")
        pass

