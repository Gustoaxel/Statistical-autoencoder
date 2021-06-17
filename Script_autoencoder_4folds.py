# -*- coding: utf-8 -*-
"""
Copyright   I3S CNRS UCA 

This code is an implementation of statistical evaluation of our autoencoder discribe in the article :
Efficient diagnostic using the latent space ofa Non-Parametric Supervised Autoencoderfor 
metabolomics datasets

When using this code , please cite Barlaud, M., Guyard, F.: Learning sparse deep neural networks 
using efficient structured projections on convex constraints for green ai. ICPR 2020 Milan Italy (2020)

and 

Axel Gustovic, Celine Ocelli, Thierry Pourcher and Michel Barlaud : Efficient diagnostic using the 
latent space ofa Non-Parametric Supervised Autoencoderfor metabolomics datasets

Params : 
    
    - Seed (line 54)
    - Database (line 81)
    - Projection (line 119)
    - ETA (line 123)
    - Standardization (line 140)
    
    
"""
import os
import sys
if '../functions/' not in sys.path:
    sys.path.append('../functions/')


import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from torch import nn 
import time
from sklearn import metrics
# lib in '../functions/'
import functions.functions_torch as ft
import functions.functions_network_pytorch as fnp
from sklearn.metrics import precision_recall_fscore_support
import analyzer.model_analyzer as ma
import seaborn as sns

  

#################################
  
if __name__=='__main__':
#------------ Parameters ---------
   
    # Set seed
    Seed = [4, 5, 6]

    
    
    # Set device (Gpu or cpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    nfold = 4
    N_EPOCHS = 10
    N_EPOCHS_MASKGRAD = 10      # number of epochs for trainning masked graident
    # learning rate 
    LR = 0.0005      
    BATCH_SIZE=8
    LOSS_LAMBDA = 0.0005         # Total loss =λ * loss_autoencoder +  loss_classification
    bW=1 # Kernel size for distributions
    

    criterion_reconstruction = nn.SmoothL1Loss(  reduction='sum'  ) # SmoothL1Loss
    
    # Loss functions for classification
    criterion_classification = nn.CrossEntropyLoss( reduction='sum'   )
    

    

    TIRO_FORMAT = True
    file_name = 'LUNG.csv'
#    file_name = 'COVID.csv'
#    file_name = "BRAIN_MID.csv"


    
    # Choose Net 
#    net_name = 'LeNet'
    net_name = 'netBio'
    n_hidden = 96  # nombre de neurone sur la couche du netBio

    # Save Results or not
    SAVE_FILE = True
    # Output Path 
    outputPath =  'results/'+ file_name.split('.')[0] + '/'
    if not os.path.exists(outputPath): # make the directory if it does not exist
        os.makedirs(outputPath)
        
    # Do pca or t-SNE 
    Do_pca = True
    Do_tSNE = True
    run_model= 'No_proj' 
    # Do projection at the middle layer or not
    DO_PROJ_middle = False

      
    # Do projection (True)  or not (False)
#    GRADIENT_MASK = False
    GRADIENT_MASK = True
    if GRADIENT_MASK:
        
        run_model='ProjectionLastEpoch'
    # Choose projection function
    if not GRADIENT_MASK:
        TYPE_PROJ = 'No_proj'
        TYPE_PROJ_NAME = 'No_proj'
    else:
#        TYPE_PROJ = ft.proj_l1ball         # projection l1
        TYPE_PROJ = ft.proj_l11ball        #original projection l11 (les colonnes a zero)
#        TYPE_PROJ = ft.proj_l21ball        # projection l21
        TYPE_PROJ_NAME = TYPE_PROJ.__name__
        
    ETA = 75  
    AXIS = 0          #  for PGL21
    
     ####### Set of parameters : #######
    # Lung : ETA = 75 Seed = [4, 5, 6]
    # Brain : ETA = 25 Seed = [4, 5, 6]
    # Covid : ETA = 200 Seed = [4, 5, 6]
  
    # Top genes params 

#    DoTopGenes = True
    DoTopGenes = False

#    PERFORM_MA = True
    PERFORM_MA = False
    
    # Saceling 
    doScale = True 
#    doScale = False
    

          
#------------ Main loop ---------
    # Load data    

    X,Y,feature_name,label_name, patient_name, LFC_Rank = ft.ReadData(file_name, TIRO_FORMAT=TIRO_FORMAT, doScale = doScale) # Load files datas
    
    LFC_Rank.to_csv(outputPath+'/LFC_rank.csv')
        
    feature_len = len(feature_name)
    class_len = len(label_name)
    print('Number of feature: {}, Number of class: {}'.format(feature_len,class_len ))

    accuracy_train = np.zeros((nfold*len(Seed),class_len+1))
    accuracy_test = np.zeros((nfold*len(Seed),class_len+1))
    data_train = np.zeros((nfold*len(Seed),7))
    data_test = np.zeros((nfold*len(Seed),7))
    correct_prediction = []
    s=0
    for SEED in Seed : 
        
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        for i in range(nfold) : 
            
            train_dl, test_dl, train_len, test_len, Ytest  = ft.CrossVal(X,Y, patient_name, BATCH_SIZE,i , SEED)
            print('Len of train set: {}, Len of test set:: {}'.format(train_len,test_len)) 
            print('----------- Début iteration ',i,'----------------')
            # Define the SEED to fix the initial parameters 
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            
            # run AutoEncoder
            if net_name == 'LeNet':
                    net = ft.LeNet_300_100(n_inputs=feature_len, n_outputs=class_len).to(device)        # LeNet  
            if net_name == 'netBio':
                    net = ft.netBio(feature_len, class_len , n_hidden).to(device)       # netBio  
         
            weights_entry,spasity_w_entry = fnp.weights_and_sparsity(net)
            topGenesCol_entry = ft.selectf(net.state_dict()['encoder.0.weight'] , feature_name)
            
            if GRADIENT_MASK:
                run_model='ProjectionLastEpoch'
        
            optimizer = torch.optim.Adam(net.parameters(), lr= LR )
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 150, gamma = 0.1)
            data_encoder, data_decoded, epoch_loss, best_test, net = ft.RunAutoEncoder(net, criterion_reconstruction, optimizer, lr_scheduler, train_dl, train_len, test_dl, test_len, N_EPOCHS, \
                        outputPath, SAVE_FILE,  DO_PROJ_middle, run_model, criterion_classification, LOSS_LAMBDA, feature_name, TYPE_PROJ, ETA, AXIS= AXIS )  
            labelpredict = data_encoder[:,:-1].max(1)[1].cpu().numpy()
            # Do masked gradient
            
            
            if GRADIENT_MASK:
                print("\n--------Running with masked gradient-----")
                print("-----------------------")
                zero_list = []
                tol = 1.0e-3
                for index,param in enumerate(list(net.parameters())):
                    if index<len(list(net.parameters()))/2-2 and index%2==0:
                        ind_zero = torch.where(torch.abs(param)<tol)
                        zero_list.append(ind_zero)
                
                # Get initial network and set zeros      
                # Recall the SEED to get the initial parameters
                np.random.seed(SEED)
                torch.manual_seed(SEED)
                torch.cuda.manual_seed(SEED)
                
                # run AutoEncoder
                if net_name == 'LeNet':
                    net = ft.LeNet_300_100(n_inputs=feature_len, n_outputs=class_len).to(device)        # LeNet   
                if net_name == 'netBio':
                    net = ft.netBio(feature_len, class_len , n_hidden).to(device)       # FairNet 
                optimizer = torch.optim.Adam(net.parameters(), lr= LR)
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 150,gamma=0.1)
                
                for index,param in enumerate(list(net.parameters())):
                    if index<len(list(net.parameters()))/2-2 and index%2==0:
                        param.data[zero_list[int(index/2)]] =0 
                        
                run_model = 'MaskGrad'
                data_encoder, data_decoded, epoch_loss, best_test, net = ft.RunAutoEncoder(net, criterion_reconstruction, optimizer, lr_scheduler, train_dl, train_len, test_dl, test_len, N_EPOCHS_MASKGRAD, \
                        outputPath,  SAVE_FILE,  zero_list, run_model, criterion_classification, LOSS_LAMBDA, feature_name, TYPE_PROJ, ETA,AXIS=AXIS )    
                print("\n--------Finised masked gradient-----")
                print("-----------------------")
            #np.save(file_name.split('.')[0]+'_Loss_'+str(run_model), epoch_loss)
            
            data_encoder = data_encoder.cpu().detach().numpy() 
            data_decoded =  data_decoded.cpu().detach().numpy() 
            
            data_encoder_test, data_decoded_test, class_train, class_test , topGenesCol, correct_pred, softmax, Ytrue, Ypred  = ft.runBestNet(train_dl, test_dl, best_test, outputPath , i , class_len , net, feature_name , test_len)
          #  data_encoder_test, data_decoded_test, class_train, class_test , topGenesCol, correct_pred = ft.runBestNet(train_dl, test_dl, best_test, outputPath , i , class_len , net, feature_name , test_len)
            
            
            
            
    
            if SEED == Seed[-1]:
                if i == 0 : 
                    Ytruef = Ytrue
                    Ypredf = Ypred
                    LP_test = data_encoder_test.numpy()
                else : 
                    Ytruef = np.concatenate((Ytruef, Ytrue))
                    Ypredf = np.concatenate((Ypredf, Ypred))
                    LP_test = np.concatenate((LP_test, data_encoder_test.numpy()))
            
            accuracy_train[s*4 + i] = class_train
            accuracy_test[s*4 + i] = class_test
            # silhouette score
            X_encoder = data_encoder[:,:-1]
            labels_encoder = data_encoder[:,-1]
            data_encoder_test = data_encoder_test.cpu().detach()
            
        
            data_train[s*4 + i,0] = metrics.silhouette_score(X_encoder, labels_encoder, metric='euclidean')
            
    
            X_encodertest = data_encoder_test[:,:-1]
            labels_encodertest = data_encoder_test[:,-1]
            data_test[s*4 + i,0]  = metrics.silhouette_score(X_encodertest, labels_encodertest, metric='euclidean')        
            # ARI score
            
            data_train[s*4 + i,1]  = metrics.adjusted_rand_score(labels_encoder, labelpredict)    
            data_test[s*4 + i,1] = metrics.adjusted_rand_score(Ytest, data_encoder_test[:,:-1].max(1)[1].numpy())
            
            
            
            # AMI Score 
            data_train[s*4 + i,2]  = metrics.adjusted_mutual_info_score(labels_encoder, labelpredict)
            data_test[s*4 + i,2] = metrics.adjusted_mutual_info_score(Ytest,data_encoder_test[:,:-1].max(1)[1].numpy() )
    
            #UAC Score 
            if class_len == 2 : 
                data_train[s*4 + i,3]  = metrics.roc_auc_score(labels_encoder, labelpredict)
                data_test[s*4 + i,3] = metrics.roc_auc_score(Ytest,data_encoder_test[:,:-1].max(1)[1].numpy() )
            
            # F1 precision recal 
            data_train[s*4 + i,4:] = precision_recall_fscore_support(labels_encoder, labelpredict, average='macro')[:-1]
            data_test[s*4 + i,4:] = precision_recall_fscore_support(Ytest,data_encoder_test[:,:-1].max(1)[1].numpy() , average='macro')[:-1]
            
           # Recupération des labels corects
            correct_prediction += correct_pred 
    
            # Get Top Genes of each class 
    
   #         method = 'Shap'       # (SHapley Additive exPlanation) A nb_samples should be define
            nb_samples =300        # Randomly choose nb_samples to calculate their Shap Value, time vs nb_samples seems exponential 
    #        method = 'Captum_ig'   # Integrated Gradients
            method = 'Captum_dl'  # Deeplift
    #        method = 'Captum_gs'  # GradientShap
            
            if DoTopGenes:
                tps1 = time.perf_counter()
                if i == 0 :
                    print("Running topGenes..." )
                    df_topGenes = ft.topGenes(X,Y,feature_name,class_len, feature_len, method, nb_samples, device ,net)
                    df_topGenes.index = df_topGenes.iloc[:,0]
                    print("topGenes finished" )
                    tps2 = time.perf_counter()
                else : 
                    print("Running topGenes..." )
                    df_topGenes = ft.topGenes(X,Y,feature_name,class_len, feature_len, method, nb_samples, device, net)
                    print("topGenes finished" )
                    df = pd.read_csv('{}{}_topGenes_{}_{}.csv'.format(outputPath,str(TYPE_PROJ_NAME),method,str(nb_samples) ),sep=';' , header = 0 , index_col = 0 )
                    df_topGenes.index = df_topGenes.iloc[:,0]
                    df_topGenes = df.join(df_topGenes.iloc[:,1] , lsuffix='_' , )
                    
                df_topGenes.to_csv('{}{}_topGenes_{}_{}.csv'.format(outputPath,str(TYPE_PROJ_NAME),method,str(nb_samples) ),sep=';')
                tps2 = time.perf_counter()
                print("execution time topGenes  : ", tps2 - tps1)
            
        if SEED == Seed[0] : 
                df_softmax = softmax
                df_softmax.index = df_softmax["Name"]
                softmax.to_csv('{}softmax.csv'.format(outputPath),sep=';',index=0)
        else : 
                softmax.index = softmax["Name"]
                df_softmax = df_softmax.join(softmax , rsuffix="_")
                
        # Moyenne sur les SEED 
        if DoTopGenes : 
            df = pd.read_csv('{}{}_topGenes_{}_{}.csv'.format(outputPath,str(TYPE_PROJ_NAME),method,str(nb_samples) ),sep=';' , header = 0 , index_col = 0 )
            df_val = df.values[1:,1:].astype(float) 
            df_mean = df_val.mean(axis = 1).reshape(-1,1)
            df_std = df_val.std(axis = 1).reshape(-1,1)
            df = pd.DataFrame(np.concatenate((df.values[1:,:],df_mean , df_std), axis = 1), columns = ["Features", "Fold 1","Fold 2","Fold 3","Fold 4","Mean","Std" ])
            df_topGenes = df 
            df_topGenes = df_topGenes.sort_values(by = "Mean", ascending = False)
            df_topGenes = df_topGenes.reindex(columns = ["Features", "Mean", "Fold 1","Fold 2","Fold 3","Fold 4","Std" ] )
            df_topGenes.to_csv('{}{}_topGenes_{}_{}.csv'.format(outputPath,str(TYPE_PROJ_NAME),method,str(nb_samples) ),sep=';' , index=0) 
        
            if SEED == Seed[0] :
                df_topGenes_mean = df_topGenes.iloc[:, 0:2]
                df_topGenes_mean.index = df_topGenes.iloc[:,0]
            else : 
                df = pd.read_csv('{}{}_topGenes_Mean_{}_{}.csv'.format(outputPath,str(TYPE_PROJ_NAME),method,str(nb_samples) ),sep=';' , header = 0 , index_col = 0 )
                df_topGenes.index = df_topGenes.iloc[:,0]
                df_topGenes_mean = df.join(df_topGenes.iloc[:,1] , lsuffix='_' , )
                    
            df_topGenes_mean.to_csv('{}{}_topGenes_Mean_{}_{}.csv'.format(outputPath,str(TYPE_PROJ_NAME),method,str(nb_samples) ),sep=';')
    
        s+= 1
        
    if class_len ==2 : 
        df_softmax = df_softmax.drop(["Name", "Name_", "Labels_", "Proba class 1", "Proba class 1_" ], axis = 1)
        df_softmax.columns = ["Labels", "1", "2", "3"]
        
        for l in range(df_softmax.shape[0]):
            if df_softmax.iloc[l,0] == 1:
                df_softmax.iloc[l,:] = df_softmax.iloc[l,:].where(df_softmax.iloc[l,1:] > 0.5)
            else:
                df_softmax.iloc[l,:] = df_softmax.iloc[l,:].where(df_softmax.iloc[l,1:] < 0.5)
        df_softmax = df_softmax.dropna(how='all')

        
    try : 
        df = pd.read_csv('{}Labelspred_softmax.csv'.format(outputPath),sep=';', header = 0 )
        data_pd = pd.read_csv('datas/FAIR/'+ str(file_name[:-12])+ ".csv",delimiter=';', decimal=",", header=0, encoding = 'ISO-8859-1')
    except : 
        data_pd = pd.read_csv('datas/FAIR/'+ str(file_name),delimiter=';', decimal=",", header=0, encoding = 'ISO-8859-1')
            
    proba = df.values[:,2:].astype(float)
    
    df.index = df.iloc[:,0]
    df = df.join(data_pd.T , rsuffix='_' , how = 'right')
    df.iloc[: , 1:4].to_csv('{}Labelspred_softmax.csv'.format(outputPath),sep=';')
    
    
    df_accTrain, df_acctest = ft.showClassResult(accuracy_train, accuracy_test, nfold*len(Seed), label_name)
    df_metricsTrain, df_metricstest = ft.showMetricsResult(data_train, data_test, nfold*len(Seed))
    # print sparsity  
    #print('\n best test accuracy:',best_test/float(test_len))
    
    # Reconstruction by using the centers in laten space and datas after interpellation
    center_mean,  center_distance = ft.Reconstruction(0.2, data_encoder, net, class_len )
              
    # Do pca,tSNE for encoder data
    if Do_pca and Do_tSNE:
        tit = "Latent Space"
        ft.ShowPcaTsne(X, Y, data_encoder, center_distance, class_len , tit )
        tit = "Latent Space Test"
        #ft.ShowPcaTsne(X, Y, data_encoder_test.numpy(), center_distance, class_len , tit)
        #ft.ShowPcaTsne(X, Y, LP_test, center_distance, class_len , tit)
    
    # Do Implementation of Metropolis and pass to decoder for reconstruction
    if DoTopGenes:
        df = pd.read_csv('{}{}_topGenes_Mean_{}_{}.csv'.format(outputPath,str(TYPE_PROJ_NAME),method,str(nb_samples) ),sep=';' , header = 0 , index_col = 0 )
        df_val = df.values[:,1:].astype(float) 
        df_mean = df_val.mean(axis = 1).reshape(-1,1)
        df_std = df_val.std(axis = 1).reshape(-1,1)
        df_meanstd = df_std/df_mean
        col_seed = ["Seed "+ str(i) for i in Seed]
        df = pd.DataFrame(np.concatenate((df.values[:,:],df_mean , df_std , df_meanstd), axis = 1), columns = ["Features"] + col_seed + ["Mean", "Std" , "Mstd"])
        df_topGenes = df 
        df_topGenes = df_topGenes.sort_values(by = "Mean", ascending = False)
        df_topGenes = df_topGenes.reindex(columns = ["Features", "Mean"] +  col_seed +["Std" , "Mstd"] )
        df_topGenes.to_csv('{}{}_topGenes_Mean_{}_{}.csv'.format(outputPath,str(TYPE_PROJ_NAME),method,str(nb_samples) ),sep=';' , index=0) 
        
    plt.figure()
    plt.title("Kernel Density")   
    plt.plot([0.5,0.5,0.5], [-1, 0,3])
    lab = 0 
    for col in softmax.iloc[:,2:] :
        distrib = softmax[col].where(softmax['Labels']== lab).dropna()
        if lab == 100 : 
            sns.kdeplot(1-distrib,bw=0.05,shade = True )
        else :
            sns.kdeplot(distrib, bw=0.1, shade=True)
            fvlist = distrib.where(distrib < 0.5).dropna()
            #print(len(fvlist)/len(softmax['Labels']))

        #sns.kdeplot(distrib,bw=0.1, shade=True)
        #sns.displot(distrib,kind=kde, bw_adjust=1 ,fill=True)
        lab += 1
    # get weights and spasity
    spasity_percentage_entry = {}
    for keys in spasity_w_entry.keys():
        spasity_percentage_entry[keys]= spasity_w_entry[keys]*100
    print('spasity % of all layers entry \n',spasity_percentage_entry)
    print("-----------------------")
    weights,spasity_w = fnp.weights_and_sparsity(net.encoder)
    spasity_percentage = {}
    for keys in spasity_w.keys():
        spasity_percentage[keys]= spasity_w[keys]*100
    print('spasity % of all layers \n',spasity_percentage)
    print("-----------------------")
    
    weights_decoder,spasity_w_decoder = fnp.weights_and_sparsity(net.decoder)
    mat_in =   net.state_dict()['encoder.0.weight']        

    mat_col_sparsity = ft.sparsity_col(mat_in, device = device)
    print(" Colonnes sparsity sur la matrice d'entrée: \n",mat_col_sparsity )
    mat_in_sparsity = ft.sparsity_line(mat_in, device = device)
    print(" ligne sparsity sur la matrice d'entrée: \n",mat_in_sparsity )
    layer_list = [x for x in weights.values() ]
    layer_list_decoder = [x for x in weights_decoder.values() ]
    titile_list = [x for x in spasity_w.keys()]
    ft.show_img(layer_list,layer_list_decoder, file_name)
    
    #plt.figure()
    #plt.title("Confusion matrix")
    #cm = confusion_matrix(Ytruef, Ypredf)
    #sns.heatmap(cm , annot=True, cmap="YlGnBu")

    #df_correct = pd.read_csv('datas/FAIR/'+ str(file_name),delimiter=';', decimal=",", header=0, encoding = 'ISO-8859-1')
    #df_correct = df_correct.loc[: , ["Name"] + correct_prediction]

    #df_correct.to_csv(outputPath+ str(file_name)[:-4]+"_correct.csv",sep=';', decimal=",", index = 0)
    
    # Model Analyzer
    # Make sure the net for "MA.set_model(net)" you used is the final net (by return the final 'net' after runNet() ) 
    if PERFORM_MA:
        print("Running Model Analyzer..." )
        MA = ma.Model_Analyzer()     
        MA.set_model(net)
        df_ma = MA.summary(feature_len, class_len,  eps= 0.001, rounding = 1)
        print("Model Analyzer finished, result is variable 'df_ma' " )
        df_ma.to_csv('{}{}_modelAnalyzer.csv'.format(outputPath,str(TYPE_PROJ_NAME) ),\
                     header=True, index=False, sep=";",decimal=",")
            
    # Loss figure
    if os.path.exists(file_name.split('.')[0]+'_Loss_No_proj.npy') and os.path.exists(file_name.split('.')[0]+'_Loss_MaskGrad.npy'):
        loss_no_proj = np.load(file_name.split('.')[0]+'_Loss_No_proj.npy')
        loss_with_proj = np.load(file_name.split('.')[0]+'_Loss_MaskGrad.npy')
        plt.figure()
        plt.title(file_name.split('.')[0]+' Loss')
        plt.xlabel('Epoch')
        plt.ylabel('TotalLoss')
        plt.plot(loss_no_proj, label = 'No projection')
        plt.plot(loss_with_proj, label = 'With projection ')
        plt.legend()
        plt.show()
    if SAVE_FILE:
        df_acctest.to_csv('{}{}_acctest.csv'.format(outputPath,str(TYPE_PROJ_NAME)),sep=';') 
        #df_topGenes.to_csv('{}{}_topGenes_{}_{}.csv'.format(outputPath,str(TYPE_PROJ_NAME),method,str(nb_samples) ),sep=';')

        print("Save topGenes results to: ' {} ' ".format(outputPath) )
        