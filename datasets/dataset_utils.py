
import numpy as np
import matplotlib.pyplot as plt
import csv
import json



def index_sizes_from_event_dict(event_dictionary):
    D=len(event_dictionary.keys())
    I_ks = []
    for d in range(D):
        I_ks.append( len(event_dictionary[d]) )
    I_ks=tuple(I_ks)
    return I_ks

def get_cumulative_index_sizes(I_ks):
    D=len(I_ks)
    cum_n = 0
    cum_I_ks = [0,]
    for d in range(D):
        cum_n += I_ks[d]
        cum_I_ks.append(cum_n)
    return cum_I_ks





def read_categorical_only_from_CSV(CSV_FILENAME,I_ks,N,event_dictionary,verbose=False):
    D=len(I_ks)
    ONEHOT_D = sum(list(I_ks))
    if verbose:
        print(I_ks)
        print(ONEHOT_D)
    X_arr = np.zeros( (N,ONEHOT_D), dtype=int )
    
    with open(CSV_FILENAME, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        n=0
        for row in spamreader:
            cum_n = 0
            for d in range(D):
                value = row[d]
                if value not in event_dictionary[d]:
                    if verbose:
                        print('\t',d)
                v_id = event_dictionary[d].index(value)
        
                X_arr[n,cum_n+v_id] = 1
                cum_n += I_ks[d]
            n+=1
    if verbose:
        print(n)
    return X_arr


def create_relevant_plot_categorical(X_arr,event_dictionary,I_ks,cum_I_ks):
    N = X_arr.shape[0]
    
    def plot_categorical(i,j):
        I_i = I_ks[i]
        I_j = I_ks[j]
        cum_I_i = cum_I_ks[i]
        cum_I_j = cum_I_ks[j]
        noise1 = np.random.randn(N) * 0.1
        noise2 = np.random.randn(N) * 0.1
        plt.scatter( np.matmul(X_arr[:,cum_I_i:cum_I_i+I_i],np.arange(I_i))+noise1,
                     np.matmul(X_arr[:,cum_I_j:cum_I_j+I_j],np.arange(I_j))+noise2,)
        plt.xticks(np.arange(I_i),event_dictionary[i])
        plt.yticks(np.arange(I_j),event_dictionary[j])
        plt.show()

    return plot_categorical


def save_onehotified_dataset_to_numpy(save_path,onehotified_array,N,D,I_ks,cum_I_ks,readable_labels,event_dictionary,trntst_split=(0.70,0.30)):
    metadata_dictionary = {}
    metadata_dictionary["N"] = N
    metadata_dictionary["D"] = D
    metadata_dictionary["I_ks"] = I_ks
    metadata_dictionary["cum_I_ks"] = cum_I_ks

    TRNVAL_N = int(N*trntst_split[0])
    TST_N = N-TRNVAL_N
    metadata_dictionary["TRNVAL_N"] = TRNVAL_N
    metadata_dictionary["TST_N"] = TST_N

    np.random.seed(0)
    perm = np.random.permutation(N)
    trn_X = onehotified_array[perm[:TRNVAL_N]]
    tst_X = onehotified_array[perm[TRNVAL_N:]]
    pass
    
    metadata_dictionary["readable_labels"]  = readable_labels
    metadata_dictionary["event_dictionary"] = event_dictionary

    #save all files
    with open(save_path+'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata_dictionary, f, indent=4)
    np.save(save_path+'X_trn.npy',trn_X)
    np.save(save_path+'X_tst.npy',tst_X)
    return trn_X,tst_X







def load_triple_from_path(dataset_id_path):
    X_trn = np.load(dataset_id_path+"X_trn.npy")
    X_tst = np.load(dataset_id_path+"X_tst.npy")
    with open(dataset_id_path+'metadata.json', 'r', encoding='utf-8') as f:
        metadata_dict = json.load(f)
    return X_trn,X_tst,metadata_dict

def load_dataset(dataset_id,dataset_path="datasets/preprocessed_data/"):

    if dataset_id=="mushroom10D_":
        X_trn,X_tst,metadata_dict = load_triple_from_path(dataset_path+"mushroom_")
    else:
        X_trn,X_tst,metadata_dict = load_triple_from_path(dataset_path+dataset_id)
        
        
    I_ks = tuple(metadata_dict['I_ks'])
    cum_I_ks = (metadata_dict['cum_I_ks'])
    D = metadata_dict['D']
    N = metadata_dict['TRNVAL_N']
        
        
    if "mushroom" in dataset_id:  #243,799,621,632,000
        
        mushroom_classes = [0,5,22] #poison,odor,habitat
        classes_to_predict = mushroom_classes
        
        if "10D" in dataset_id:
            D2 = 10            #829,440
            #D2 = 16    #10,749,542,400
            classes_to_predict=classes_to_predict[:2]
            
            I_ks = list(I_ks)[:D2]
            cum_I_ks = list(cum_I_ks)[:D2]
            cum_D2 = sum(I_ks)
            I_ks = tuple(I_ks)
            D=D2
            X_trn = X_trn[:,:cum_D2]
            X_tst = X_tst[:,:cum_D2]
            
    elif "breastcancer" in dataset_id: # 598752
        #classes_to_predict = [] #TODO
        classes_to_predict = [0,6] #10/08/2024 @ 8:30pm PST 
        
    elif "adults" in dataset_id:  # 3495260160000 = 3.5 trillion
        adult_classes = [0,1,8,9] #income,age,race,gender
        classes_to_predict = adult_classes
         
        if "V2" in dataset_id: # 650280960000 = 0.65 trillion
            pass

        
    return X_trn,X_tst, I_ks,cum_I_ks,D,N,classes_to_predict















