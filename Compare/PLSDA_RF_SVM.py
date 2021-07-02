# -*- coding: utf-8 -*-
"""
Copyright   I3S CNRS UCA 

This code is an implementation of the other methods used for comparison in the article :
Efficient diagnostic using the latent space ofa Non-Parametric Supervised Autoencoderfor 
metabolomics datasets

Params : 
    
    - Seed (line 32)
    - Database (line 31)
    - Standardization (line 45)
    - Algorithme to compare (line 34)
    - Features extraction (line 36)
    


"""
import functions as ff
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale as scale
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from random import randrange
import random


if __name__ == '__main__':
    
    # Set params : 
    filename = 'LUNG.csv'
    Seed = [6, 7]
    alglist = ['plsda', 'RF' ] # Other ML algorithm to compare
    doTopgenes = True         # Features selection 

    
    
    # Load data 
    X, Yr, nbr_clusters, feature_names = ff.readData(filename)

    # Data Preprocessiong
    #X=normalize(X,norm='l1',axis=1)
    X=np.log(abs(X+1))
    X=X-np.mean(X,axis=0)
    X=scale(X,axis=0)
    #X=scale(X,axis=1)
    X=X/ff.normest(X)
    
    
    
    ######## Main ####### 
    
    print("Starts trainning")
    for i in Seed:
        # Processing
        print("------ Seed {} ------".format(i))
        accTestCompare,df_timeElapsed, aucTestCompare =\
        ff.basic_run_other(
                       X,Yr,nbr_clusters,alglist,
                       genenames=None,
                       clusternames=None,
                       nfold=4,
                       rng=6,
                       outputPath='../results/')
        if doTopgenes : 
            df_featureList = ff.rankFeatures(X,Yr,alglist,feature_names)


    
        
        if i == Seed[0] : 
            accTestCompare_final = accTestCompare.iloc[:4, :]
            aucTestCompare_final = aucTestCompare.iloc[:4, :]
            if doTopgenes:
                df_featureList_final = df_featureList
        else : 
            accTestCompare_final= pd.concat([accTestCompare_final , accTestCompare.iloc[:4, :]])
            aucTestCompare_final= pd.concat([aucTestCompare_final , aucTestCompare.iloc[:4, :]])
            if doTopgenes:
                for met in range(len(df_featureList_final)):
                    df_featureList_final[met] = df_featureList_final[met].join(df_featureList[met]["weights"], rsuffix=" {}".format(i))
    mean = pd.DataFrame(accTestCompare_final.mean(axis = 0))
    if doTopgenes:
        for met in range(len(df_featureList_final)) : 
            mean_met = pd.DataFrame(df_featureList_final[met].iloc[:,1:].mean(axis = 1))
            std_met = pd.DataFrame(df_featureList_final[met].iloc[:,1:].std(axis = 1))
            mean_met.columns= ["Mean"]
            df_featureList_final[met] = df_featureList_final[met].join(mean_met)
            std_met.columns= ["Std"]
            df_featureList_final[met] = df_featureList_final[met].join(std_met)
    
    std = pd.DataFrame(accTestCompare_final.std(axis = 0))
    mean.columns= ["Mean"]
    accTestCompare_final = accTestCompare_final.T.join(mean).T
    std.columns= ["Std"]
    accTestCompare_final = accTestCompare_final.T.join(std).T
    
    mean = pd.DataFrame(aucTestCompare_final.mean(axis = 0))
    std = pd.DataFrame(aucTestCompare_final.std(axis = 0))
    mean.columns= ["Mean"]
    aucTestCompare_final = aucTestCompare_final.T.join(mean).T
    std.columns= ["Std"]
    aucTestCompare_final = aucTestCompare_final.T.join(std).T
    
    
    
    color = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD','#8C564B', '#E377C2', '#BCBD22', '#17BECF', '#40004B','#762A83',\
             '#9970AB', '#C2A5CF', '#E7D4E8', '#F7F7F7','#D9F0D3', '#A6DBA0', '#5AAE61', '#1B7837', '#00441B','#8DD3C7', '#FFFFB3',\
             '#BEBADA', '#FB8072', '#80B1D3','#FDB462', '#B3DE69', '#FCCDE5', '#D9D9D9', '#BC80BD','#CCEBC5', '#FFED6F']


    random.seed(Seed[0])
    lab = np.unique(Yr)
    index = [[] for l in lab ]
    test_index = []
    train_index = []
    for i in range(len(Yr)) :
            for l in lab : 
                if l == Yr[i] : 
                    index[l-1].append(i)
    for l in index : 
            test_index.append( l.pop(randrange(len(l))))
            train_index += l
    print(" test index = ", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Yr[train_index], Yr[test_index]

    clf = PLSRegression(n_components=4,scale=False)
    model = clf.fit(X_train,y_train.ravel())
    col = [color[i] for i in y_train ]
    plt.figure()
    plt.scatter(model.x_scores_[:,0], model.x_scores_[:,1], c=col )
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Score plot PLSDA on BRAIN")
    
    testm = model.transform(X_test)
    col = [color[i+4] for i in y_test ]
    plt.scatter(testm[:,0], testm[:,1], c=col , s=120, marker='s')
    plt.show()
    
    

    
    
    
