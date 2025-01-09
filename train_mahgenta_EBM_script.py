#!/usr/bin/env python
# coding: utf-8





from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("masking_loss_variant", None, "['pseudo','unmasked','masked',]")
flags.DEFINE_boolean("MIS_search_or_many_body",True,"True=MIS.False=Boltz")

flags.DEFINE_integer("gpu_id", 0, "cuda index of the GPU")
flags.DEFINE_integer("epochs", 10, "number of epochs per each round (total when doing many-body)")
flags.DEFINE_integer("trnval_shuffle_seed", 1, "seed before permutation to split trn and val")

flags.DEFINE_string("dataset_prefix_identifier", None, "['mushroom_','mushroom10D_','adults_','adultsV2_','breastcancer_',]")
flags.DEFINE_boolean("saving_results", True, "Save the results of the trained model.")




flags.DEFINE_string("AIS_type_of_on", 'end_of_epochs', "['off','end_of_epochs','every_step','every_ten',]")
flags.DEFINE_integer("num_of_gibbs_steps", 3, "number of gibbs steps")
flags.DEFINE_integer("num_of_MIS_steps", 310, "number of steps of the MIS algorithm to take")
flags.DEFINE_integer("interaction_batch_size", 10, "number of interactions to add at each step of the MIS algorithm")




### MIS==TRUE case
flags.DEFINE_string("heredity_strength", None, "['strong100','semistrong','weak50','weak30',]")
flags.DEFINE_boolean("MIS_renorm_ON_or_OFF", False, "True=ON.False=OFF.")
### MIS==FALSE case
flags.DEFINE_integer("many_body_index", 1, "one/two/three -body.")



import os
import time
import copy
import json
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader


import sys
from models import LowBodyLegendre_LogLinearGAM


from utils.utils import *
from datasets.dataset_utils import load_dataset
from utils.MIS_utils import complete_MIS_search
    








    
def main(argv):
    del argv

    print(FLAGS)
    for thing in FLAGS:
        print(thing,FLAGS.flag_values_dict()[thing])
    
    
    MIS_search_is_ON = FLAGS.MIS_search_or_many_body
    if MIS_search_is_ON:
        assert (FLAGS.heredity_strength is not None)
        assert (FLAGS.MIS_renorm_ON_or_OFF is not None)
    else:
        assert (FLAGS.many_body_index is not None)



    datetimestr = gettimestamp()
    print(datetimestr)



    SAVING = FLAGS.saving_results
    SAVING_PATH = "results_data/"
    gpu_id = FLAGS.gpu_id
    the_device = torch.device('cuda:'+str(gpu_id))


    
    GRADUAL_TRAINING = True
    masking_loss_variant_string = FLAGS.masking_loss_variant
    if masking_loss_variant_string=='pseudo':
        NULL_MASKING,PSEUDOLIKELIHOOD_OFF = False,False   #PSUEDO
    elif masking_loss_variant_string=='unmasked':
        NULL_MASKING,PSEUDOLIKELIHOOD_OFF = False,True    #NULL STILL OFF
    elif masking_loss_variant_string=='masked':
        NULL_MASKING,PSEUDOLIKELIHOOD_OFF = True,True   #TRUE LIKE
    else:
        raise Exception('invalid masking/loss variant FLAG')

        
        
        
        
        
    ### GIBBS HYPERPARAMETERS ###
    GIBBS_HYPERPARAMETERS = { 
        'full_zbs' : 100*1000,
        'num_of_steps' : FLAGS.num_of_gibbs_steps,
        'type_of_steps' : 'block_mixedNewPartialRounds_byCount',
            'partial_amount' : 10,
    }
    
    if not MIS_search_is_ON:
        GIBBS_HYPERPARAMETERS = {
            'full_zbs' : 100*1000,
            'num_of_steps' : FLAGS.num_of_gibbs_steps,

            'type_of_steps' : 'block_partialRounds_byPerc',
                'partial_perc' : 0.10
        #         'partial_perc' : 0.50
        }


    GIBBS_ON = False
    GIBBS_ON = True



    RUIN_SPLIT_THETA_NAUGHTS = False
    RUIN_SPLIT_THETA_NAUGHTS = True
    if not GIBBS_ON:
        RUIN_SPLIT_THETA_NAUGHTS=False


    #technically no eyes... (cant calculate renormalization constant)
    #gradient updates are only at the will of the Gibbs samples
    #original name was "LOOK_NO_HANDS"
    GRADIENT_UPDATES_WITHOUT_RENORMALIZATION_CONST = False
    GRADIENT_UPDATES_WITHOUT_RENORMALIZATION_CONST = True  


    if not GIBBS_ON and GRADIENT_UPDATES_WITHOUT_RENORMALIZATION_CONST:
        print("WARNING: NEVER DO THIS! PLEASE LOOK AND PUT YOUR HANDS BACK ON!")
        raise Exception('you gotta look bro')






    dataset_id = FLAGS.dataset_prefix_identifier
    X_arr_trnval,___,I_ks,cum_I_ks,D,TRNVAL_N,classes_to_predict = load_dataset(dataset_id)
    print('X_arr_trnval',X_arr_trnval.shape)
    print("N,D\t",TRNVAL_N,D)

    ENTROPY_CALC_MODE = "X_default"
    # ENTROPY_CALC_MODE = "P_full_tensor"


    #AIS HYPERPARAMETERS
    K_temp = 40 
    AIS_gibbs_steps = 1
    AIS_partial_amount = 40
    M_samples = 10*1000

    temperatures = np.linspace(0,1,K_temp+1)
    AIS_ON = FLAGS.AIS_type_of_on
    AIS_HYPERPARAMETERS = {
        'M_samples' : M_samples,
        'AIS_gibbs_steps' : AIS_gibbs_steps,
            'AIS_partial_amount' : AIS_partial_amount,
            'K_temp' : K_temp,
    } 





    T=FLAGS.num_of_MIS_steps   #usually 300 or so
    INT_BS = FLAGS.interaction_batch_size


    HEREDITY_STR = FLAGS.heredity_strength
    PARAM_RENORMALIZATION = FLAGS.MIS_renorm_ON_or_OFF
    MANY_BODY_INDEX = FLAGS.many_body_index


    if MIS_search_is_ON: #mahgenta using MIS algorithm (precomputable path with MMI score)
        frontiers = complete_MIS_search(X_arr_trnval,D,I_ks,cum_I_ks,  T=T,
                                        PARAM_RENORM=PARAM_RENORMALIZATION,INT_BS=INT_BS,HEREDITY_STRENGTH=HEREDITY_STR)


        print("FINAL COUNT OF SELECTED")
        my_inters = frontiers[-1][0]
        print(len(my_inters))

        frontiers=frontiers[:302]
        print("FINAL COUNT OF SELECTED")
        my_inters = frontiers[-1][0]
        print(len(my_inters))


    
    
    else: #1-body, 2-body BOLTZMANN EXPERIMENTS
        # ### [1,2,3]-body instead of MIS

        the_singles = [(i,) for i in range(D)]
        print(the_singles)
        the_doubles = []
        for i in range(D):
            for j in range(i+1,D):
                the_doubles.append( (i,j) )
        print(the_doubles)
        the_triples = []
        for i in range(D):
            for j in range(i+1,D):
                for k in range(j+1,D):
                    the_triples.append( (i,j,k) )
        print(the_triples)
        print()

        the_doubles = sorted(the_doubles,key=lambda x: sum(list(x)))
        the_triples = sorted(the_triples,key=lambda x: sum(list(x)))

        frontiers  = []
        my_inters = [()]
        
        my_inters.extend(the_singles)
        if MANY_BODY_INDEX==1:
            frontiers.append( (my_inters,None) )
        my_inters = copy.deepcopy(my_inters)
        my_inters.extend(the_doubles)
        if MANY_BODY_INDEX==2:
            frontiers.append( (my_inters,None) )
        my_inters = copy.deepcopy(my_inters)
        my_inters.extend(the_triples)
        if MANY_BODY_INDEX==3:
            frontiers.append( (my_inters,None) )

        print("FINAL COUNT OF SELECTED")
        my_inters = frontiers[-1][0]
        print(len(my_inters))


    trnval_split = (0.70,0.30)
    TRN_N = int(TRNVAL_N*trnval_split[0])
    VAL_N = TRNVAL_N - TRN_N
    print('TRN_N',TRN_N,'VAL_N',VAL_N)

    np.random.seed(1)
    np.random.seed(FLAGS.trnval_shuffle_seed)
    perm2 = np.random.permutation(TRNVAL_N) 
    X_arr_trn = X_arr_trnval[perm2[:TRN_N]]
    X_arr_val = X_arr_trnval[perm2[TRN_N:]]





    BS = X_arr_trn.shape[0]

    if ENTROPY_CALC_MODE=="X_default":
        print('X_arr_trn',X_arr_trn.shape)
        X_trn_torch = torch.from_numpy(X_arr_trn).float().to(the_device)
        trn_loader = DataLoader(dataset=X_trn_torch, batch_size=BS,shuffle=True)
        num_batches = len(trn_loader)
        print("BS",BS)
        print('batches',num_batches)
    else:
        raise Exception("not reimplemented yet")


    X_trn = X_arr_trn
    X_val = X_arr_val

    mom = 0.5
    LR_ANNEALING = False
    LR_ANNEALING = True
    LR = 1e0
    LR = 5e-1
    
    EP = FLAGS.epochs
    verbose = True


    time_takens=[]
    AIS_time_takens=[]
    Gibbs_time_takens=[]


    p=0
    tensor_of_KLs = np.zeros((len(frontiers),  1,EP+1,7))
    tensor_of_theta_over_time = np.zeros((len(frontiers),  EP+1,1))
    tensor_of_thetaX_over_time = np.zeros((len(frontiers),  EP+1,6))
    tensor_of_Accs= np.zeros((len(frontiers),  1,EP+1,len(classes_to_predict)))
    tensor_of_CEs = np.zeros((len(frontiers),  1,EP+1,len(classes_to_predict)))

    
    
    compare_CE_base,compare_CE_perf = baseEnt_across_mushroom_directions_X(torch.Tensor(X_arr_trnval), I_ks,classes_to_predict)



    lastly_trained_model = None



    if SAVING:
        experiment_type_str = "mahgenta_"
        if not MIS_search_is_ON:
            experiment_type_str = "manybody_"
        if not os.path.exists(SAVING_PATH):
            os.makedirs(SAVING_PATH)
        saving_model_file_prefix = SAVING_PATH+dataset_id+experiment_type_str+datetimestr+"__"

        hyperparameters_dict = {"LR":LR,"EP":EP,'NULL_MASKING':NULL_MASKING,'GRADUAL_TRAINING':GRADUAL_TRAINING,
                                'len(frontiers)':len(frontiers),
                                'masking_loss_variant_string' : masking_loss_variant_string,
                                'HEREDITY_STR' : HEREDITY_STR,
                                'PARAM_RENORMALIZATION' : PARAM_RENORMALIZATION,
                                'INT_BS' : INT_BS,
                                'MANY_BODY_INDEX' : MANY_BODY_INDEX,
                                "GIBBS_ON":GIBBS_ON,"GIBBS_HYPERPARAMETERS":GIBBS_HYPERPARAMETERS,
                                "AIS_ON":AIS_ON,"AIS_HYPERPARAMETERS":AIS_HYPERPARAMETERS}
        #SAVE AS A JSON
        with open(saving_model_file_prefix+'hyperparameter.json', 'w', encoding='utf-8') as f:
            json.dump(hyperparameters_dict, f, ensure_ascii=False, indent=4)
            
        #need to be saving this as well because yeah otherwise it is hard to match afterwards
        with open(saving_model_file_prefix+'frontiers.json', 'w', encoding='utf-8') as f:
            json.dump(frontiers, f, ensure_ascii=False, indent=4)





    LR_OG = LR
    for ff,front in enumerate(frontiers):
        print('ff',ff)
        real_start_time = time.time()
        my_inters, my_index_dictionary = front
        LR = LR_OG

        start_time = time.time()

        theta_param_model = LowBodyLegendre_LogLinearGAM(my_inters,I_ks).to(the_device)




        if ff==0 and MIS_search_is_ON:
            theta_param_model = LowBodyLegendre_LogLinearGAM(my_inters,I_ks).to(the_device)
            if True: 
                unif_naught = np.log(  theta_param_model.get_cumprod()  )
                with torch.no_grad():
                    theta_param_model.theta_parameters["()"] -= (unif_naught)
            lastly_trained_model = copy.deepcopy(theta_param_model)
            lastly_trained_model = lastly_trained_model.cpu()

            if NULL_MASKING:
                theta_param_model.add_the_masks_to_thetas(torch.from_numpy(X_arr_trnval).float().to(the_device))

            if GIBBS_ON:
                w_gibbs = theta_param_model.generate_uniformReplacing_z_batch(full_zbs=GIBBS_HYPERPARAMETERS['full_zbs'])

            if ENTROPY_CALC_MODE == "P_full_tensor":
                theta_param_model.X_or_P_mode = "P";
            print("theta_param_model.X_or_P_mode",theta_param_model.X_or_P_mode)
        else:
            
            
            if ff==0 and not MIS_search_is_ON: 
                no_inters = [()]
                theta_param_model = LowBodyLegendre_LogLinearGAM(no_inters,I_ks).to(the_device)
                theta_param_model.precompute_maximal_splitting()     
                theta_param_model.precompute_all_valid_hemi_indices() 
                theta_param_model.precompute_all_split_hemi_taus()    
                if True: 
                    unif_naught = np.log(  theta_param_model.get_cumprod()  )
                    with torch.no_grad():
                        theta_param_model.theta_parameters["()"] -= (unif_naught)
                lastly_trained_model = copy.deepcopy(theta_param_model)
                lastly_trained_model = lastly_trained_model.cpu()
                
                if NULL_MASKING:
                    lastly_trained_model.add_the_masks_to_thetas(torch.from_numpy(X_arr_trnval).float().to(the_device))
                    
                if GIBBS_ON:
                    w_gibbs = theta_param_model.generate_uniformReplacing_z_batch(full_zbs=GIBBS_HYPERPARAMETERS['full_zbs'])
                if ENTROPY_CALC_MODE == "P_full_tensor":
                    theta_param_model.X_or_P_mode = "P";
                    
                    
                
            if GRADUAL_TRAINING:
                x_trnval = torch.from_numpy(X_arr_trnval).float().to(the_device)
                theta_param_model=copy.deepcopy(lastly_trained_model).to(the_device)
                old_inters = theta_param_model.my_inters
                new_inters = [tup for tup in my_inters if tup not in old_inters]
                print('new_inters',new_inters)
                for new_tup in new_inters:
                    print('\t new_tup '+str(new_tup))
                    if not GRADIENT_UPDATES_WITHOUT_RENORMALIZATION_CONST:
                        if NULL_MASKING:
                            theta_param_model.add_new_interaction_tuple(new_tup,x_trnval)
                        else:
                            theta_param_model.add_new_interaction_tuple(new_tup,x_trnval,recompute_masks=False,recompute_hemis=True)


                    else:
                        if NULL_MASKING:
                            theta_param_model.add_new_interaction_tuple(new_tup,x_trnval,recompute_masks=True,recompute_hemis=False)
                        else:
                            theta_param_model.add_new_interaction_tuple(new_tup,x_trnval,recompute_masks=False,recompute_hemis=False)

                    if GIBBS_ON:
                        w_gibbs = theta_param_model.one_step_of_block_Gibbs_updates(w_gibbs,new_tup) 
                if GIBBS_ON:
                    theta_param_model.new_tups = new_inters
                    print('theta_param_model.new_tups',theta_param_model.new_tups)
                pass
            else:
                print("NOT WORKING RIGHT NOW IF YOU TURN OFF GRADUAL TRAINING")


        opt = torch.optim.SGD(theta_param_model.parameters(), lr = LR, momentum=mom)

        if True: 
            if not GRADUAL_TRAINING and not NULL_MASKING:
                pass
                print("(?) also not working right now")
            else:
                with torch.no_grad():
                    if theta_param_model.all_good_idx_tensor_list is None:
                        if ff==0: 
                            theta_param_model.precompute_maximal_splitting()                       

                    if not GRADIENT_UPDATES_WITHOUT_RENORMALIZATION_CONST:
                        theta_param_model.recenter_theta_naught_for_split_model()



        list_of_KLs = np.zeros((1,EP+1,7))

        if True:
            list_of_KLs[p,0,0] = KL_of_trusted_model(torch.Tensor(X_arr_trn).to(the_device),theta_param_model)
            list_of_KLs[p,0,2] = KL_of_trusted_model(torch.Tensor(X_arr_val).to(the_device),theta_param_model)
            print('at the start',list_of_KLs[p,0,0],list_of_KLs[p,0,2])
        tensor_of_Accs[ff,p,0,:] = Acc_across_mushroom_directions_X(torch.Tensor(X_arr_val).to(the_device),theta_param_model,classes_to_predict)
        tensor_of_CEs[ff,p,0,:] = crossEnt_across_mushroom_directions_X(torch.Tensor(X_arr_val).to(the_device),theta_param_model,classes_to_predict)




        for k in range(EP):
            ep_start_time = time.time()
            cum_gibbs_time = 0.0
            if verbose:
                print('k',k)
            for j,x_batch in enumerate(trn_loader):
                if verbose:
                    print('\tj',j)
                pass



                if True:
                    if not GIBBS_ON:
                        if PSEUDOLIKELIHOOD_OFF:
                            theta_param_model.compute_custom_backwards_split_flattened_etas(x_batch)
                        else:
                            theta_param_model.compute_pseudolikelihood_custom_backwards_split_flattened_etas_X(x_batch)

                    else:
                        if PSEUDOLIKELIHOOD_OFF:
                            theta_param_model.compute_custom_backwards_split_flattened_etas_utilizing_gibbs_X(x_batch,w_gibbs)
                        else:
                            raise Exception("not implemented yet (gibbs w/ psuedo)")

                if True:
                    if not GRADIENT_UPDATES_WITHOUT_RENORMALIZATION_CONST:
                        theta_param_model.purify_theta_gradients()
                    else:
                        theta_param_model.purify_theta_gradients_semiUnif_butMaskZero_vf()


                lambda_1 = None
                lambda_1 = 1e-3
                if (lambda_1 is not None):
                    L12_loss = torch.zeros(1).to(the_device)
                    theta_two_norms = theta_param_model.get_group_lasso_norms()
                    for tup in theta_param_model.my_inters:
                        if tup!=():
                            L12_loss += lambda_1 * theta_two_norms[tup]
                    L12_loss.backward()


                opt.step()
                opt.zero_grad()


                if GIBBS_ON: #need to update the gibbs samples
                    gibbs_start_time=time.time()
                    gibbs_steps = GIBBS_HYPERPARAMETERS['num_of_steps']
                    for gibby in range(gibbs_steps):
                        print('   gibby',gibby)

                        gibbs_type = GIBBS_HYPERPARAMETERS['type_of_steps']
                        if gibbs_type=="coordinate_rounds":
                            w_gibbs = theta_param_model.one_round_of_Gibbs_updates(w_gibbs)
                        elif gibbs_type=="block_rounds":
                            w_gibbs = theta_param_model.one_round_of_block_Gibbs_updates(w_gibbs) 
                        elif gibbs_type=="block_steps":
                            w_gibbs = theta_param_model.random_block_Gibbs_updates(w_gibbs,amount=1) 
                        elif gibbs_type=="block_partialRounds_byPerc":
                            partial_perc = GIBBS_HYPERPARAMETERS['partial_perc']
                            w_gibbs = theta_param_model.one_partialRound_of_block_Gibbs_updates(w_gibbs,perc=partial_perc)
                        elif gibbs_type=="block_partialRounds_byCount":
                            partial_amount = GIBBS_HYPERPARAMETERS['partial_amount']
                            w_gibbs = theta_param_model.one_partialRound_of_block_Gibbs_updates(w_gibbs,amount=partial_amount) 

                        elif gibbs_type=="block_mixedNewPartialRounds_byCount":
                            partial_amount = GIBBS_HYPERPARAMETERS['partial_amount']
                            w_gibbs = theta_param_model.one_partialNewRound_of_block_Gibbs_updates(w_gibbs) 
                            w_gibbs = theta_param_model.one_partialRound_of_block_Gibbs_updates(w_gibbs,amount=partial_amount) 

                        else:
                            print("UNSUPPORTED GIBBS SAMPLING TYPE")


                    gibbs_time_taken = time.time()-gibbs_start_time
                    print('Gibbs_time_taken_ff',gibbs_time_taken)
                    Gibbs_time_takens.append(gibbs_time_taken)


                if True:
                    with torch.no_grad():
                        if RUIN_SPLIT_THETA_NAUGHTS:
                            temp_split_theta_naughts = theta_param_model.split_theta_naughts

                        if not GRADIENT_UPDATES_WITHOUT_RENORMALIZATION_CONST:
                            theta_param_model.recenter_theta_naught_for_split_model() #also updates the theta_naught global

                        if RUIN_SPLIT_THETA_NAUGHTS:
                            theta_param_model.split_theta_naughts = temp_split_theta_naughts

                        theta_naught = theta_param_model.theta_parameters[str(())].data.detach().cpu()
                        print('theta_naught',theta_naught)
                        tensor_of_theta_over_time[ff,k+1] = theta_naught




            # --- finished with one epoch; updating validation statistics --- #
            if not GRADIENT_UPDATES_WITHOUT_RENORMALIZATION_CONST:
                list_of_KLs[p,k+1,0] = KL_of_trusted_model(torch.Tensor(X_arr_trn).to(the_device),theta_param_model)
                list_of_KLs[p,k+1,2] = KL_of_trusted_model(torch.Tensor(X_arr_val).to(the_device),theta_param_model)
            else:
                list_of_KLs[p,k+1,4] = JS_of_empirical_distributions_X(torch.Tensor(X_arr_trn).to(the_device),w_gibbs)
                list_of_KLs[p,k+1,5] = JS_of_empirical_distributions_X(torch.Tensor(X_arr_val).to(the_device),w_gibbs)

                AIS_start_time_ff = time.time()
                if ((AIS_ON=='off') and False) or ((AIS_ON=='end_of_epochs') and k+1==EP) or \
                   ((AIS_ON=='every_step') and True) or ((AIS_ON=='every_ten') and (k+1)%10==0):
                    
                    
                    temperatures = np.linspace(0,1,K_temp+1)

                    w_ais_gibbs = theta_param_model.generate_uniformReplacing_z_batch(full_zbs=M_samples)
                    if NULL_MASKING:
                        w_ais_gibbs = theta_param_model.generate_hemiUniformReplacing_z_batch(full_zbs=M_samples)     
                    log_scale_ratios = torch.zeros( M_samples )
                    log_scale_rat_tensor = torch.zeros( (K_temp,M_samples))

                    AIS_start_time=time.time()
                    with torch.no_grad():
                        for kk,temp in enumerate(temperatures[1:]):
                            print(' (AIS)',kk,temp)
                            old_cold = temperatures[kk]
                            new_cold = temperatures[kk+1]  

                            log_q = theta_param_model.forward_log_batch(w_ais_gibbs)
                            log_scale_ratios += (new_cold-old_cold) * log_q.cpu()
                            log_scale_rat_tensor[kk] = log_q.cpu()

                            for ais_gibby in range(AIS_gibbs_steps): 
                                w_ais_gibbs = theta_param_model.one_partialRound_of_block_Gibbs_updates(w_ais_gibbs,amount=AIS_partial_amount, temp=new_cold) 
                        AIS_time_taken = time.time() - AIS_start_time
                        print("AIS_time_taken",AIS_time_taken)

                        final_guess_logZ = torch.logsumexp(log_scale_ratios,dim=0) - np.log(M_samples)
                        unif_naught = np.log(  theta_param_model.get_cumprod()  )
                        
                        gap = (final_guess_logZ-(-unif_naught))
                        print('final_guess_logZ',final_guess_logZ)
                        print('final_gap_guess',gap)
                        theta_param_model.theta_parameters["()"] -= gap
                        xdd =  unif_naught + torch.logsumexp(log_scale_rat_tensor,dim=-1) - np.log(M_samples)
                AIS_time_taken_ff = time.time()-AIS_start_time_ff
                print('AIS_time_taken_ff',AIS_time_taken_ff)
                AIS_time_takens.append(AIS_time_taken_ff)



                if True:  
                    list_of_KLs[p,k+1,0] = KL_of_trusted_model(torch.Tensor(X_arr_trn).to(the_device),theta_param_model)
                    list_of_KLs[p,k+1,2] = KL_of_trusted_model(torch.Tensor(X_arr_val).to(the_device),theta_param_model)

            tensor_of_Accs[ff,p,k+1,:] = Acc_across_mushroom_directions_X(torch.Tensor(X_arr_val).to(the_device),theta_param_model,classes_to_predict)
            tensor_of_CEs[ff,p,k+1,:] = crossEnt_across_mushroom_directions_X(torch.Tensor(X_arr_val).to(the_device),theta_param_model,classes_to_predict)


            if verbose:
                print("  KL(P,Q) - forward  -",list_of_KLs[p,k+1,0])
                print("  KL(Q,P) - backward -",list_of_KLs[p,k+1,1])
                print(" 'KL(P,Q) - forward  -",list_of_KLs[p,k+1,2])
                print(" 'KL(Q,P) - backward -",list_of_KLs[p,k+1,3])

                print("  JS(P,Q) - forward  -",list_of_KLs[p,k+1,4])
                print("  JS(Q,P) - backward -",list_of_KLs[p,k+1,5])

                xd = (compare_CE_base-tensor_of_CEs[ff,p,k+1,:])/(compare_CE_base-compare_CE_perf)
                print(" 'CE(P;Q) - classwise-",xd)
                print("'acc(P;Q) - classwise-",tensor_of_Accs[ff,p,k+1,:])
                print('seconds',time.time()-ep_start_time)
            if LR_ANNEALING: #0.98^100 ~= 0.13
                LR = LR * 0.98
                for g in opt.param_groups:
                    g['lr'] = LR

        total_time = time.time()-start_time
        real_total_time = time.time()-real_start_time
        if True:
            print('total_time',total_time)
            print('real_total_time',real_total_time)
            if GIBBS_ON:
                print('gibbs_total_time',gibbs_time_taken)
            if AIS_ON in ['end_of_epochs','every_step']:
                print('AIS_time_taken_ff',AIS_time_taken_ff)


        tensor_of_KLs[ff] = list_of_KLs
        time_takens.append(total_time)

        if GRADUAL_TRAINING:
            lastly_trained_model = copy.deepcopy(theta_param_model.cpu())  

        if SAVING:
            torch.save(theta_param_model.cpu(),saving_model_file_prefix+"frontier_"+str(ff)+".pt")
            np.save(saving_model_file_prefix+"tensor_of_KLs.npy",tensor_of_KLs)
            np.save(saving_model_file_prefix+"tensor_of_Accs.npy",tensor_of_Accs)
            np.save(saving_model_file_prefix+"tensor_of_CEs.npy",tensor_of_CEs)
            with open(saving_model_file_prefix+'time_takens.txt','w') as f:
                f.write('time_takens'+str(time_takens)+'\n')
                f.write('AIS_time_takens'+str(AIS_time_takens)+'\n')
    pass


    print('tensor_of_KLs',tensor_of_KLs.shape)
    for ff,front in enumerate(frontiers):
        xd = tensor_of_KLs[ff,0,-1,:]
        print(xd[0],'\t',xd[2])


    print(sum(time_takens)/60,'minutes')
    print()
    print(time_takens)



if __name__ == '__main__':
    app.run(main)

