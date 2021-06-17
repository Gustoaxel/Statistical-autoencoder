'''
Created on Oct 10, 2019

@author: yfgi6212
'''

from __future__ import print_function, division

import os
import sys
import math
import copy
#import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import collections

    
if '../models/' not in sys.path:
    sys.path.append('../models/')
    


import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import scipy.stats as ss


from analyzer.layer_utils import *

DEFAULT_BASE_MODEL = "../results/MNIST/net_epoch-20_Adagrad_lr-0.01.pkl"
DEFAULT_MODEL = "../results/MNIST/net_epoch-20_Proximal_PGL1_gamma-0.02_eta-50_CS.pkl"
DEFAULT_CIFAR = "../results/CIFAR10/net_epoch-100_Adam_lr-0.01.pkl"
DEFAULT_CIFAR_PGL1 = "../results/CIFAR10/net_epoch-100_Proximal_PGL1_gamma-0.01_eta-1500_CS.pkl"

DEFAULT_LAYER_TYPES =  [
                        torch.nn.modules.activation.ReLU,
                        torch.nn.modules.conv.Conv2d,
                        torch.nn.modules.pooling.MaxPool2d,
                        torch.nn.modules.pooling.AdaptiveAvgPool2d,
                        torch.nn.modules.linear.Linear,
                        torch.nn.BatchNorm2d,
                        torch.nn.Dropout2d,
                        torch.nn.Flatten
                        ]

EPSILON = 0.001 # truncation for the sparsity 
FLOAT_ROUNDING = 4



class Model_Analyzer:
    """Class for analyzing models structure
       
    """
    
    def __init__(self,model = None):
        self._model = model
        if not(model is None):
            self._structure = self._struct()
        else:
            self._structure   = None
        self._structured_data = None
        self._input_list     = list()
        
    # Properties
    
    @property
    def model(self):
        return(self._model)
        
    @property
    def structure(self):
        return(self._structure)
    
    
    
    # methods
    def load_model(self,f):
        if issubclass(type(f),nn.Module):
            self._model = f.cpu()
        else:
            with open(f, 'rb') as f:
                params = torch.load(f,map_location=torch.device('cpu'))
                model = params['model'].cpu()
                State_dict = collections.OrderedDict()
    
                for k, v in params['state_dict'].items():
                    name = k.replace('module.','') # remove `module.`
                    State_dict[name] = v
    
                model.load_state_dict(State_dict)
    
            self._model = model.cpu()
            
        self._structure = self._struct()
            
        
    def set_model(self,model):
        self._model = model
        self._structure = self._struct()
            
    def save_model(self,file):
        torch.save({"model":self._model})
        
    def _struct(self):
        mo = self._model.modules()
        
        admissible = tuple([k for k in DEFAULT_LAYER_TYPES])
        structure = {}
        layer_index = -1
        for index,layer in enumerate(mo):
        
            if issubclass(type(layer),admissible):
                layer_index += 1
                structure[layer_index] = layer
        return(structure)
        
    def _layer_representation(self,layer,c_in,h_in,w_in):
        
        pad = pading(layer)
        dil = dilation(layer)
        st  = stride(layer)
        ks  = kernel_size(layer)
        
        
        output = output_size(c_in,h_in,w_in,pad,dil,ks,st)
        
        Ox  = output[0]
        Oy  = output[1]
        
        if (issubclass(type(layer),torch.nn.modules.conv.Conv1d) 
            or issubclass(type(layer),torch.nn.modules.conv.Conv2d)
            or is_linear(layer)):
                c_out = channels_out(layer)
        else:
            c_out = c_in
        
        return([c_out,Ox,Oy])
    
    def model_representation(self,in_shape): # in_shape=[1,1] if the first layer is linear!
        """ Model representation 
        
        input: h_in,w_in 2d shape of input
        output: dataframe: layer, real_in, real_out, channel_in, channel_out
        """
        
        
        h_in = in_shape[0]
        w_in = in_shape[1]
        
         
        Struct = self._struct()
        
        if is_linear(Struct[0]):
            Input_shape = [1,Struct[0].in_features]
        else:
            Input_shape = [1,Struct[0].in_channels(),h_in,w_in]
        
        
        
        X = torch.from_numpy(np.random.random(Input_shape)).float()
        
        Struct = self._struct()
        
        Rep = []
        
        
        for k in Struct.keys():
                   
            #####
           
             
            In = [int(torch.cumprod(torch.tensor(X.shape),dim=0)[-1]),X.shape[1] if len(list(X.shape))==4 else 1]
            
             
            if not (isinstance(Struct[k],torch.nn.modules.batchnorm.BatchNorm2d)
                    or (isinstance(Struct[k],torch.nn.modules.batchnorm.BatchNorm1d))):
                
                X = Struct[k](X)
    
           
            Out = [int(torch.cumprod(torch.tensor(X.shape),dim=0)[-1]),X.shape[1] if len(list(X.shape))==4 else 1]
            
            
            Output_shape = list(X.shape)
            
            if len(Output_shape)==2: # linear layer are convolutive layers with kernel (1,1)
                Output_shape = Output_shape+[1,1]
                
            if len(Input_shape)==2: # linear layer are convolutive layers with kernel (1,1)
                Input_shape = Input_shape+[1,1]
            
            Rep.append([Struct[k],In[0],Out[0],Input_shape[1],Output_shape[1],kernel_size(Struct[k])[0],kernel_size(Struct[k])[1],Input_shape,Output_shape,0,[],0,[]])
            
            Input_shape = Output_shape
            
        Rep = pd.DataFrame(data=np.array(Rep),columns=['layer','R_in','R_out','C_in','C_out','kernel_x','kernel_y','input_shape','output_shape','NVC','VC','NVR','VR'])
        
        
        return(Rep)
        
        
        
    def reduce_model(self,input_shape,epsilon=EPSILON): # in_shape=[1,1] if the first layer is linear
        
        model = self.model_representation(input_shape)
        
        layer_number = model.shape[0]
        
        Red = {}
        
        for k in range(layer_number):
            
            Red[k] = {}
            if k != 0:
                Red[k]['VR'] = vanishing_row(model.loc[k]['layer'],indices=True)
            else:
                Red[k]['VR'] = []
            if k != (layer_number - 1):
                Red[k]['VC'] = vanishing_column(model.loc[k]['layer'],indices=True)
            else:
                Red[k]['VC'] = []
            
            Red[k]['NC'] = model.loc[k]['input_shape'][1] - len(Red[k]['VC'])  # number of remaining column
            Red[k]['NR'] = model.loc[k]['output_shape'][1] - len(Red[k]['VR']) # number of remaining rows
            Red[k]['NVR'] = len(Red[k]['VR'])                                  # number of vanishing column
            Red[k]['NVC'] = len(Red[k]['VC'])                                  # number of vanishing column
           
        # boundaries
        Red[-1] = {}
        Red[-1]['VR'] = []
        Red[-1]['VC'] = []
        Red[-1]['NVR'] = len(Red[-1]['VR'])
        Red[-1]['NVC'] = len(Red[-1]['VC'])
        Red[-1]['NR'] = 0
        Red[-1]['NC'] = 0
        
        
        Red[layer_number] = {}
        Red[layer_number]['VR'] = []
        Red[layer_number]['VC'] = []
        Red[layer_number]['NVR'] = len(Red[layer_number]['VR'])
        Red[layer_number]['NVC'] = len(Red[layer_number]['VC'])
        Red[layer_number]['NR'] = 0
        Red[layer_number]['NC'] = 0
       
        
        # propagate vanishing rows
        for k in range(0,layer_number):
            
            if is_transparent(model.loc[k]['layer']):
                Red[k]['VC'] = list(set(Red[k]['VC']).union(set(Red[k-1]['VR'])))
                Red[k]['NVC'] = len(Red[k]['VC'])
                Red[k]['NC'] = model.loc[k]['input_shape'][1] - Red[k]['NVC']
                Red[k]['VR'] = Red[k]['VC']
                Red[k]['NVR'] = Red[k]['NVC']
                Red[k]['NR']  = Red[k]['NC']
                
            elif not is_flatten(model.loc[k]['layer']): # conv2d or Linear
                Red[k]['VC'] = list(set(Red[k]['VC']).union(set(Red[k-1]['VR'])))
                Red[k]['NVC'] = len(Red[k]['VC'])
                Red[k]['NC'] = model.loc[k]['input_shape'][1] - Red[k]['NVC']
                
            else: # flatten
                Red[k]['VC'] = list(set(Red[k]['VC']).union(set(Red[k-1]['VR'])))
                Red[k]['NVC'] = len(Red[k]['VC'])
                Red[k]['NC']  =  model.loc[k]['input_shape'][1] - Red[k]['NVC']


                vanish_col = transfer_indices(Red[k]['VC'],model.loc[k]['input_shape'],model.loc[k]['output_shape'])
                
                Red[k]['VR'] = list(set(Red[k]['VR']).union(vanish_col))
                Red[k]['NVR'] = Red[k]['VR']
                Red[k]['NR'] = model.loc[k]['output_shape'][1] - len(Red[k]['VR'])

        
#        # propagate vanishing cols
        for k in range(layer_number-1,-1,-1):
            
            
            if is_transparent(model.loc[k]['layer']):
                Red[k]['VR'] = list(set(Red[k]['VR']).union(set(Red[k+1]['VC'])))
                Red[k]['NVR'] = len(Red[k]['VR'])
                Red[k]['NR'] = model.loc[k]['output_shape'][1] - len(Red[k]['VR'])
                Red[k]['VC'] = list(set(Red[k]['VC']).union(set(Red[k]['VR'])))
                Red[k]['NVC'] = len(Red[k]['VC'])
                Red[k]['NC'] = model.loc[k]['input_shape'][1] - Red[k]['NVC']
                
            elif not is_flatten(model.loc[k]['layer']):
                Red[k]['VR'] = list(set(Red[k]['VR']).union(set(Red[k+1]['VC'])))
                Red[k]['NVR'] = len(Red[k]['VR'])
                Red[k]['NR'] = model.loc[k]['output_shape'][1] - len(Red[k]['VR'])
            else:    
                
                Red[k]['VR'] = list(set(Red[k]['VR']).union(set(Red[k+1]['VC'])))
                Red[k]['NVR'] = len(Red[k]['VR'])
                Red[k]['NR'] = model.loc[k]['output_shape'][1] - Red[k]['NVR']
                
                vanish_row = transfer_indices(Red[k]['VR'],model.loc[k]['output_shape'],model.loc[k]['input_shape'])
                
                Red[k]['VC'] = list(set(Red[k]['VC']).union(set(vanish_row)))
                Red[k]['NVC'] = len(Red[k]['VC'])
                Red[k]['NC']  = model.loc[k]['input_shape'][1] - Red[k]['NVC']
            
            Red[k]['NR'] = model.loc[k]['output_shape'][1] - len(Red[k]['VR'])  # number of remaining column
            Red[k]['NVR'] = len(Red[k]['VR'])
            
            
        MRed = []
        # format the output as the one of model_representation
        for k in range(0,layer_number):
            
            layer = model.loc[k]['layer']
            VC    = Red[k]['VC']
            C_in   = Red[k]['NC']
            NVC    = Red[k]['NVC']
            VR    = Red[k]['VR']
            NVR   = Red[k]['NVR']
            C_out    = Red[k]['NR']
            kernel_x = model.loc[k]['kernel_x']
            kernel_y = model.loc[k]['kernel_y']
            R_in     = model.loc[k]['input_shape'][2]*model.loc[k]['input_shape'][3]*C_in
            input_shape = [1,C_in,model.loc[k]['input_shape'][2],model.loc[k]['input_shape'][3]]
            R_out    = model.loc[k]['output_shape'][2]*model.loc[k]['output_shape'][3]*C_out
            output_shape = [1,C_out,model.loc[k]['output_shape'][2],model.loc[k]['output_shape'][3]]
            
            MRed.append([layer,R_in,R_out,C_in,C_out,kernel_x,kernel_y,input_shape,output_shape,NVC,VC,NVR,VR])
            
            
        Red = pd.DataFrame(data=np.array(MRed),columns=['layer','R_in','R_out','C_in','C_out','kernel_x','kernel_y','input_shape','output_shape','NVC','VC','NVR','VR'])
        
        return(Red)
                
        
            
    def summary(self,h_in,w_in, propagate=True, eps=EPSILON, rounding = FLOAT_ROUNDING): 
        """
        - h_in: horizontal size of the input image
        - w_in: vertical size of the input image
        """
        
    
        reduced = self.reduce_model([h_in,w_in])
        reduced_perf = model_performance(reduced,propagate=propagate)
        
        

        Total = []
        Total.append("Total")
        Total.append("-")
        Total.append("-")
        Total.append(reduced_perf['param'].astype(int).sum())
        Total.append(reduced_perf['maccs'].astype(int).sum())
        Total.append("-")
        Total.append("-")
        Total.append(reduced_perf['bytes16'].astype(int).sum())
        Total.append(reduced_perf['bytes32'].astype(int).sum())
        Total.append(reduced_perf['0_coeffs'].astype(int).sum())
        Total.append(reduced_perf['0_row'].astype(int).sum())
        Total.append(reduced_perf['0_col'].astype(int).sum())
        Total.append("-")
        Total.append("-")

        reduced_perf.loc[reduced_perf.shape[0]] = Total

        
        return(reduced_perf)

# Usefull methods using ModelAnalyzer

def transfer_indices(index,shape_1,shape_2):
    
    if shape_1 == shape_2:
        Ind = index
    else:
        X = np.ones(shape_1)
    
        M = np.zeros((shape_1[2],shape_1[3]))
    
        for i in index:
            X[:,i] = M

        XT = X.reshape(shape_2)        
    
        XT = np.array([[np.sum(XT[:,i]) for i in range(shape_2[1])]])
    
        Ind = list(np.where(XT == 0)[1])
    return(Ind)
    
    
    
    
    
def opaque(layer):
    
    return(is_conv(layer) or is_linear(layer))


# evaluation of MACCS for the layer representation
def maccs(layer_rep):
    
    macc = 0
    
    if is_linear(layer_rep['layer']):
        macc = layer_rep['C_in']*layer_rep['C_out']
        
    elif is_conv(layer_rep['layer']):
        
        macc = layer_rep['C_in']*layer_rep['C_out']*layer_rep['kernel_x']*layer_rep['kernel_y']*layer_rep['output_shape'][2]*layer_rep['output_shape'][3]
        
    return(macc)

# number of parameters of the layer
# input should be the full representation row
def param(layer_rep):
    
    param = 0
    
#    if is_linear(layer_rep[0]):
#        param = layer_rep['C_in']*layer_rep['C_out'] + layer_rep['C_out']
#    elif is_conv(layer_rep[0]):
#        param = layer_rep['C_in']*layer_rep['C_out']*layer_rep['kernel_x']*layer_rep['kernel_y']+ layer_rep['C_out']


    if (is_linear(layer_rep['layer']) or is_conv(layer_rep['layer'])):
        param = layer_rep['C_in']*layer_rep['C_out']*layer_rep['kernel_x']*layer_rep['kernel_y']
        
    return(param)

def model_performance(rep,propagate=True,eps=EPSILON,rounding=FLOAT_ROUNDING):
    """
    Add entropy16, entropy32 and MACCS to a given model representation
    
    Args:
        -rep: model representation - output of either model_representation of forward_reduction
        
    Output:
        - a model representation with 
              MACCs,entropy16, entropy32, #params, #memory_bytes16,#memory_Bytes32
          appended to the layer representation
    """
    
    perf_rep = []
    
    
    for k in range(rep.shape[0]):
        
        MACCS = maccs(rep.loc[k])
        
        if is_presentable(rep.loc[k]['layer']):
            
            if propagate:
                weight = restrict_weight(rep.loc[k])
            else:
                weight = rep.loc[k]['layer'].weight.cpu().data.numpy()
            
            
            
            
            zeros = np.zeros(weight.shape)
            weights = np.where(np.abs(weight)>eps,weight,zeros)
            nweights = np.size(weights)
            params    = param(rep.loc[k])
            entropy32 = entropy(normalize_weight(rep.loc[k]),dtype=32,rounding = rounding)
            
            #print(f"entropy 32 {entropy32}, n_weight: {np.size(normalized_weight)}")
            nbytes32  = math.ceil(entropy32*params/8.)
                 
            entropy16 = entropy(normalize_weight(rep.loc[k]),dtype=16,rounding=rounding)
            #print(f"entropy 16 {entropy16}, n_weight: {np.size(normalized_weight)}")
            nbytes16  = math.ceil(entropy16*params/8.)
            params    = param(rep.loc[k])
            zero_c    = zero_coeffs(rep.loc[k])
            zero_f_c  = rep.loc[k]['NVC']
            zero_f_r  = rep.loc[k]['NVR']
            l1_weight = round(l1_norm(weights),rounding)
            l1_bias   = round(l1_norm(rep.loc[k]['layer'].bias.cpu().data.numpy()),rounding)
            
            layerType = rep.loc[k]['layer'].__repr__().split('(')
            layerType = layerType[0]
            
            newperf   = [layerType,rep.loc[k]['C_in'],rep.loc[k]['C_out'],params,MACCS,entropy16,entropy32,nbytes16,nbytes32,zero_c,zero_f_r,zero_f_c,l1_weight,l1_bias]
        else:
            layerType = rep.loc[k]['layer'].__repr__().split('(')
            layerType = layerType[0]
            newperf   = [layerType,rep.loc[k]['C_in'],rep.loc[k]['C_out'],0,MACCS,0,0,0,0,0,0,0,0,0] 
            
        perf_rep.append(newperf)
        # add the estimated storage size using the source coding theorem: the entropy is the averaged bits par symbol
        # for compression. Here we use Bytes instead of bits         
        # add weight sparsity info
                 
        
        
    perf = pd.DataFrame(data = np.array(perf_rep),columns = ['layer',
                                                   'c_in',
                                                   'c_out',
                                                   'param',
                                                   'maccs',
                                                   'ent16',
                                                   'ent32',
                                                   'bytes16',
                                                   'bytes32',
                                                   '0_coeffs',
                                                   '0_row',
                                                   '0_col',
                                                   'l1(weight)',
                                                   'l1(bias)'])
        
    return(perf)
    

def restrict_weight(layer_rep):
    
    M = layer_rep['layer'].weight.cpu().data.numpy()
    
    # set vanishing rows to 0
    
    VR = layer_rep['VR']
    ZERO = np.zeros(M[0].shape)
    for k in VR:
        M[k] = ZERO
        
    # set vanishing columns to 0    
    
    VC = layer_rep['VC']
    ZERO = np.zeros(M[:,0].shape)
    for k in VC:
        M[:,k] = ZERO
        
    return(M)


def entropy(np_array,dtype=32,rounding = FLOAT_ROUNDING):
    
    if dtype==32:
        ar = np_array.astype(np.float32)
    elif dtype==16:
        ar = np_array.astype(np.float16)
    else:
        print("Not defined for dtype="+str(dtype))
        return(0)
        
    
    # unique values
    
    Keys = np.unique(ar,return_counts=True)
    
    values = Keys[0]
    distrib = Keys[1]

    distrib = distrib/distrib.sum()
    
    # calculation of entropy
    
    ent = np.sum(-distrib*np.log2(distrib))+1.0
    return(round(ent,rounding))
    
        
    
def zero_coeffs(layer_rep):
    
    
    M = layer_rep['layer'].weight.cpu().data.numpy()
    
    if is_linear(layer_rep['layer']):
        the_shape = list(M.shape)+[1,1]
        M = M.reshape(tuple(the_shape))
    else:
        the_shape = M.shape
        
    # set to 0 vanishing rows
    
    #Row = np.zeros((M.shape[1],M.shape[2],M.shape[3]))
    
    Row = np.zeros(M[0].shape)
    
    for k in layer_rep['VR']:
        M[k] = Row
        
    # set to 0 vanishing columns 
    
    #Column = np.zeros((M.shape[0],M.shape[2],M.shape[3]))
    Column = np.zeros(M[:,0].shape)
    
    for k in layer_rep['VC']:
        M[:,k] = Column
        
    return(len(np.where(M.flatten()==0)[0]))
    
    
def zero_filter_col(np_array):
    
    nval = 0
    if np_array.ndim == 1:
        nval = zero_coeffs(np_array)
    elif np_array.ndim > 1:
        nval = zero_coeffs(np.array([np.sum(np.abs(np_array[:,i])) for i in range(np_array.shape[1])]))
        
    return(nval)
    
def zero_filter_row(np_array):
    
    nval = 0
    if np_array.ndim == 1:
        nval = zero_coeffs(np_array)
    elif np_array.ndim > 1:
        nval = zero_coeffs(np.array([np.sum(np.abs(np_array[i])) for i in range(np_array.shape[0])]))
        
    return(nval)
    
def normalize_parameters(b):
    """ Suppress the vanishing rows and columns """
    
#    a = copy.deepcopy(np_array)
#    # suppress vanishing columns
#    
#    vcols_ind = np.where(np.array([np.sum(np.abs(a[:,i])) for i in range(a.shape[1])])==0)
#    vrows_ind = np.where(np.array([np.sum(np.abs(a[i])) for i in range(a.shape[0])])==0)
#   
#    b = np.delete(a,vcols_ind,1) # suppress vanishing columns
#    b = np.delete(b,vrows_ind,0) # suppress vanishing rows
    
    return(b)
    
def normalize_weight(layer_rep):
    
    weight = layer_rep['layer'].weight.cpu().data.numpy()
    
    the_shape = weight.shape
    
    if len(the_shape) == 2:
        weight = weight.reshape((the_shape[0],the_shape[1],1,1))
    
    VR = layer_rep['VR']
    VC = layer_rep['VC']
    
    
    W1 = np.delete(weight,VR,axis=0)
    W2 = np.delete(W1,VC,axis=1)
    
    
    if len(the_shape)==2:
        new_shape = W2.shape
        W2.reshape((new_shape[0],new_shape[1]))
    
    return(W2)
    
    
def global_entropy(model,dtype=32,rounding = FLOAT_ROUNDING):
    
    params = list([])
    
    for k in model.named_modules():
        
        if issubclass(type(k[1]),tuple([torch.nn.Conv2d,torch.nn.Linear])):
            params = params + list(k[1].weight.detach().cpu().numpy().reshape(-1))
            
    return(entropy(np.array(params),dtype=dtype,rounding = rounding))
    
    
def l1_norm(np_array):
    return(np.sum(np.abs(np_array)))
    
    
## FOR TESTS
    
def load_model(filename):
    
    params = torch.load(filename)
    model = params['model']
    
    State_dict = collections.OrderedDict()
    
    for k, v in params['model_state'].items():
        name = k.replace('module.','') # remove `module.`
        State_dict[name] = v
    
    model.load_state_dict(State_dict)
    
    return(model,params)
    
##
##    
##    
#if __name__ == '__main__':
#    
#    
##    MA1 = Model_Analyzer()
##    
##    Initial = "F:/Project/Gradient/Net4/results/retrain_models/Best_R_l1-10.0_Net4_MNIST_Adam_lr-0.001_epoch-[1,20]_seed-5034_version-1.pth"
##    initial_model, params = load_model(Initial)
##    MA1.load_model(initial_model)
##    Initial_Reduce = MA1.reduce_model([28,28])
##    Initial_Perf = MA1.summary(28,28,propagate=True)
#    
#    MA2 = Model_Analyzer()
#    Sparse = "F:/Project/Gradient/Net4/results/retrain_models/Best_R_l1-10.0_Net4_MNIST_Adam_lr-0.001_epoch-[1,20]_seed-5034_version-1.1.pth"
#    sparse_model, params = load_model(Sparse)
#    MA2.load_model(sparse_model)
#    Red = MA2.reduce_model([28,28])
#    Sparse_Perf = MA2.summary(28,28,propagate=True)
##    
##    
#    
#   
#    