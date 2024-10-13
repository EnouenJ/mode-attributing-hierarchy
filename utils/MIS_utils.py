import numpy as np
import copy
import time



from .utils import powerset,param_count,oneless




def constructWeakHigherInteractions(interactions_list,n,D, tau=0.5):
    next_list = []
    possible_dict = {}
    aux_interactions_list = interactions_list + [(i,) for i in range(D)]
    
    pppp = len(interactions_list)
    for i in range(pppp):
        tuple1 = interactions_list[i]
        if len(tuple1)==n:
            for j in range(i+1,len(aux_interactions_list)):
                tuple2 = aux_interactions_list[j]
                if len(tuple2)==n:

                    tuple_combined = list(tuple1)
                    for thing in tuple2:
                        if not thing in tuple_combined:
                            tuple_combined.append(thing)

                    if len(tuple_combined)==(n+1):
                        tuple_combined = tuple(sorted(tuple_combined))

                        if not tuple_combined in possible_dict:
                            score = 0
                            oneless_tuples = list(oneless(tuple_combined))
                            for subset in oneless_tuples:
                                if subset in interactions_list:
                                    score += 1
                            possible_dict[tuple_combined] = score
                            
                            if score/(n+1) >= tau:
                                next_list.append(tuple_combined)

    return next_list


def build_nonstrict_frontier(included_tuples,K,D,tau=0.5):
    nonstrict_frontier = []
    for n in range(K):
        if n==0:
            next_list = [(i,) for i in range(D)]
        else:
            next_list = constructWeakHigherInteractions(included_tuples,n,D,tau=tau)
            
        for thing in next_list:
            if not thing in included_tuples:
                nonstrict_frontier.append(thing)
                
    return nonstrict_frontier


#following the "Sp HO-BM" paper which adds a K-tuple if it is selected by (K-1) of its one-less-subsets
def build_semistrict_frontier(included_tuples,K,D,amount_less=1):
    nonstrict_frontier = []
    for n in range(K):
        if n==0:
            next_list = [(i,) for i in range(D)]
        else:
            tau_n = (n+1-amount_less)/(n+1) - 0.001
            next_list = constructWeakHigherInteractions(included_tuples,n,D,tau=tau_n)
            
        for thing in next_list:
            if not thing in included_tuples:
                nonstrict_frontier.append(thing)
                
    return nonstrict_frontier






def compute_entropy_X(X, I_ks, cum_I_ks, tup):
    K = len(tup)
    X_cum = np.ones(tuple([X.shape[0]]+[1]*K))
    for ii,i in enumerate(tup):
        cum_i = cum_I_ks[i]
        Ik_i = I_ks[i]
        X_i = X[:,cum_i:cum_i+Ik_i]

        expansion_tuple = tuple( [slice(None)] + [(None)]*ii + [slice(None)] + [(None)]*(K-1-ii) )
        X_cum = X_cum*X_i[expansion_tuple]
    P_int = np.mean(X_cum,axis=0)
    

    H_int =  (P_int[P_int!=0]*np.log(P_int[P_int!=0]))
    H_int = -np.sum(H_int)
    return H_int

def compute_entropy_X_safe(X, I_ks, cum_I_ks, tup):
    K = len(tup)
    X_cum = np.ones(tuple([X.shape[0]]+[1]*K))
    
    P_int = np.ones(1)
    if K>0:
        event_dict = {}
        for n in range(len(X)):
            current_event = []
            for ii,i in enumerate(tup):
                cum_i = cum_I_ks[i]
                Ik_i = I_ks[i]
                X_i = X[n,cum_i:cum_i+Ik_i]
                current_event.append(X_i)
            current_event = np.concatenate(current_event)
            current_event = tuple(current_event)

            if current_event in event_dict:
                event_dict[current_event] += 1
            else:
                event_dict[current_event] = 1

        P_int = np.array(list(event_dict.values()))/len(X)
        print('P_int',P_int.shape)

    H_int =  (P_int[P_int!=0]*np.log(P_int[P_int!=0]))
    H_int = -np.sum(H_int)
    return H_int


def compute_entropy_P(P, I_ks, cum_I_ks, tup):
    K = len(tup)
    D = len(I_ks)
    
    axes_to_sum = []
    for i in range(D):
        if i not in tup:
            axes_to_sum.append(i)
            
    P_int = np.ones(1)
    if tup!=():
        P_int = np.sum(P,axis=tuple(axes_to_sum)).reshape(-1)

    H_int =  (P_int[P_int!=0]*np.log(P_int[P_int!=0]))
    H_int = -np.sum(H_int)
    return H_int













def build_next_frontier(my_inters,HERED_str,K,D):
    if HERED_str=='semistrong':
        cand_list = build_semistrict_frontier(my_inters,K=K,D=D)
    else:
        if HERED_str=='weak30':
            cand_list = build_nonstrict_frontier(my_inters,K=K,D=D,tau=0.30)
        elif HERED_str=='weak50':
            cand_list = build_nonstrict_frontier(my_inters,K=K,D=D,tau=0.50)
        elif HERED_str=='strong100':
            cand_list = build_nonstrict_frontier(my_inters,K=K,D=D,tau=1.00)
    return cand_list


def complete_MIS_search(X_arr,D,I_ks,cum_I_ks,  T=100,PARAM_RENORM=False,INT_BS=10,HEREDITY_STRENGTH='weak30'):

    #MAXIMUM ORDER of the HO-interaction search procedure
    MAX_ORDER=5


    frontiers  = []
    my_inters = [()]
    MIS_start_time=time.time()

    my_ents = {}
    my_infs = {}



    for t in range(T):
        print('t',t)
        best_jouhou = None
        best_tup = None
        
        cand_list = build_next_frontier(my_inters,HEREDITY_STRENGTH,K=MAX_ORDER,D=D)
        
        
        print(cand_list)
        if len(cand_list)==0: 
            break
        for tt,tup in enumerate(cand_list):
            if tup not in my_infs:
                purepower = list(powerset(tup))
                for pp,puretup in enumerate(purepower):
                    if puretup not in my_ents:
                        my_ents[puretup]= compute_entropy_X(X_arr, I_ks, cum_I_ks, puretup)
                information=0.0
                for pp,puretup in enumerate(purepower):
                    sign = -1.0
                    if (len(puretup)-len(tup))%2 == 1:
                        sign = 1.0
                    information += sign*my_ents[puretup]
                if len(tup)==1:
                    information += np.log(I_ks[tup[0]])
                my_infs[tup] = information
            pass   
        
            jouhou = my_infs[tup] 
            if PARAM_RENORM:
                jouhou = jouhou/param_count(I_ks,tup)
                
            if best_jouhou is None:
                best_jouhou = jouhou
                best_tup = tup
            if jouhou > best_jouhou:
                best_jouhou = jouhou
                best_tup = tup
                
        to_be_added = [best_tup]
        being_added = []
        for new_tup in to_be_added:
            powerset_list = list(powerset(new_tup))
            for subset in powerset_list:
                if subset not in my_inters and subset not in being_added:
                    being_added.append(subset)
        being_added = sorted(being_added, key=lambda tup:len(tup))
        for new_tup in being_added:
            my_inters = copy.deepcopy(my_inters)
            my_inters.append( new_tup )
            
            if (len(my_inters)-1)%INT_BS == 0:
                frontiers.append( (my_inters,None) )
        print(best_tup)
        print(my_inters)
        print()
    MIS_time_taken = time.time()-MIS_start_time
    print("MIS_time_taken")
    print(MIS_time_taken/60,'minutes')
    return frontiers


