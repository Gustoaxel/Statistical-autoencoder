"""
Created on Wed Oct 09 15:20:47 2019

@author: Frédéric GUYARD
  
"""

from __future__ import print_function, division

from torch.nn.modules.module import Module
import torch.nn as nn

import os
import sys
import math
#import random
import pandas as pd
import numpy as np

#if '../functions/' not in sys.path:
#    sys.path.append('../functions/')
    
#if '../models/' not in sys.path:
#    sys.path.append('../models/')

#import functions.functions as fnp

import numpy as np
import torch
import torch.nn as nn
import scipy.stats as ss
from collections.abc import Iterable

if '../models/' not in sys.path:
    sys.path.append('../models/')
    
#import models

EPSILON = 0.001 # truncation for the sparsity 
FLOAT_ROUNDING = 4


# Method for extracting layers informations

def is_pool(layer):
    return( isinstance(layer,torch.nn.modules.pooling._MaxPoolNd)
        or isinstance(layer,torch.nn.modules.pooling._MaxUnpoolNd)
        or isinstance(layer,torch.nn.modules.pooling._AvgPoolNd)
        )
    
def is_conv(layer):
    return (issubclass(type(layer),torch.nn.modules.conv._ConvNd))
        
def is_linear(layer):
    return issubclass(type(layer),nn.Linear)


def is_channel_transformer(layer):
    return(isinstance(layer,(nn.Linear,nn.Conv2d,nn.Flatten)))
    
def is_flatten(layer):
    return(isinstance(layer,nn.Flatten))
    

def pading(layer):
    
    pad = (0,0)
    
    if (isinstance(layer,torch.nn.MaxPool1d)
       or  isinstance(layer,torch.nn.modules.conv.Conv1d)):
        pad = (layer.padding,0)
        
    elif (isinstance(layer,torch.nn.MaxPool2d)
       or  issubclass(type(layer),torch.nn.modules.conv.Conv2d)):
        
        prepad = layer.padding
        
        if isinstance(prepad,int):
            pad = (prepad,prepad)
        else:
            pad = prepad
     
    elif (is_linear(layer)):
        pad = (0,0)
    
    return(pad)
    
def dilation(layer):
    
    dil = (1,1)
    
    if (isinstance(layer,torch.nn.MaxPool1d)
       or  isinstance(layer,torch.nn.modules.conv.Conv1d)):
        dil = (layer.dilation,1)
        
    elif (isinstance(layer,torch.nn.MaxPool2d)
       or  issubclass(type(layer),nn.Conv2d)):
        
        predil = layer.dilation
        
        if isinstance(predil,int):
            dil = (predil,predil)
        else:
            dil = predil
    return(dil)
    
def stride(layer):
    
    st = (1,1)
    
    if (isinstance(layer,torch.nn.MaxPool1d)
       or  isinstance(layer,torch.nn.modules.conv.Conv1d)):
        st = (layer.stride,1)
    elif (isinstance(layer,torch.nn.MaxPool2d)
       or  issubclass(type(layer),nn.Conv2d)):
        
        prest = layer.stride
        
        if isinstance(prest,int):
            st = (prest,prest)
        else:
            st = prest
    return(st)
    
    
def kernel_size(layer):
    
    kn = (1,1)
    
    if (isinstance(layer,torch.nn.MaxPool1d)
       or  isinstance(layer,torch.nn.modules.conv.Conv1d)):
        kn = layer.kernel_size
    elif (isinstance(layer,torch.nn.MaxPool2d)
       or  issubclass(type(layer),nn.Conv2d)):
        
        prekn = layer.kernel_size
        
        if isinstance(prekn,int):
            kn = (prekn,prekn)
        else:
            kn = prekn
    return(kn)

def channels_in(layer):
    ci = 0
    
    if is_pure_conv(layer):
         ci = layer.in_channels
    elif is_linear(layer):
        ci = layer.in_features
        
    else:
        ci = 0
        
    return(ci)
    
def channels_out(layer):
    ci = 0
    
    if is_pure_conv(layer):
         ci = layer.out_channels
    elif is_linear(layer):
        ci = layer.out_features
        
    else:
        ci = 0
        
    return(ci)
        
def output_size(c_in,h_in,w_in,pad,dil,k_size,st):
    out_x = math.floor((h_in+2*pad[0]-dil[0]*(k_size[0]-1)-1)/st[0]+1)
    out_y = math.floor((h_in+2*pad[1]-dil[1]*(k_size[1]-1)-1)/st[1]+1)
    return([out_x,out_y])


def is_transparent(layer):
    
    Transparent = tuple([torch.nn.ReLU,torch.nn.Dropout3d,
                         torch.nn.Dropout2d, torch.nn.Dropout,
                         torch.nn.BatchNorm3d,torch.nn.BatchNorm2d,
                         torch.nn.BatchNorm1d,torch.nn.MaxPool2d,
                         torch.nn.MaxPool1d])
    return(issubclass(type(layer),Transparent))
    
def is_opaque(layer):
    
    return(not is_transparent(layer))
        
        
def is_pure_conv(layer):
    
    CONV = tuple([nn.Conv1d, nn.Conv2d])
    return (issubclass(type(layer),CONV))
        

def is_linear(layer):
    return issubclass(type(layer),nn.Linear)


def is_presentable(layer):
    return(is_pure_conv(layer) or is_linear(layer))


    
def vanishing_column(layer,eps=EPSILON,indices=False):

   
    
    if indices:
        vc = []
    else:
        vc = 0
    
    if is_pure_conv(layer) or is_linear(layer):
        weights = layer.weight.detach().cpu().numpy()
        
        dimensions = weights.shape
        
        nrow = dimensions[0]
        ncol = dimensions[1]
        
        
        temp = np.array([np.sum(np.abs(weights[:,i])) for i in range(ncol)])
        ind = np.where(temp<=eps)[0].tolist()
        
        if indices:
            vc = ind
        else:
            vc = len(ind)
        
    return(vc)


def vanishing_row(layer,eps=EPSILON,indices=False):

    if indices:
        vr = []
    else:
        vr = 0
    
    if is_pure_conv(layer) or is_linear(layer):
        weights = layer.weight.detach().cpu().numpy()
        
        dimensions = weights.shape
        
        nrow = dimensions[0]
        ncol = dimensions[1]
      
        temp = np.array([np.sum(np.abs(weights[i])) for i in range(nrow)])
        ind = np.where(temp<=eps)[0].tolist()
        
        if indices:
            vr = ind
        else:
            vr = len(ind)
    return(vr)
    
    
        
