import argparse
import itertools as it
import sys
import importlib
import numpy as np
import scipy as sp
from scipy import stats
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import gc

import echocABM.functions as my

class base():
    kind='base'
    #overlap=my.overlap #might turn useful?
    def __init__(self,N,K,S,e,alpha,p_I,P,m,*args):
        self.N=N
        self.K=K
        self.fractionS=S
        self.S=S*np.log(K) #bear with me
        self.e=e
        self.alpha=alpha
        self.p_I=p_I
        self.P=P
        self.m=m
        self.I=my.information(K,P,m)
        self.S_I=stats.entropy(self.I[0])
        self.S_I_norm=self.S_I/np.log2(K)
        self.v=my.generate_with_selection(N,K,self.S)
        self.center=[1/K for i in range(K)]
        self.personality=np.zeros(N)

    def factory(kind, *args):
        #if kind == "unbiased": return unbiased(*args) #alternative method
        return getattr(importlib.import_module('echocABM.model'),kind)(*args)
        
    def p_agreement(self,x,v2):
        return np.clip(float(my.overlap(self.v[x],v2))+np.random.choice([self.e,-self.e]), 0, 1)
        
    def renormalize(self,vec,l,agree,increment):
        vec=[e if i==l else e-agree*increment/(self.K-1) for i,e in enumerate(vec)]
        esclusi=[l]
        while np.amin(vec)<0:
            increment=np.abs(np.amin(vec))
            minimum=np.argmin(vec)
            esclusi.append(minimum)
            vec[minimum]=0
            vec=[e if i in esclusi else e-increment/(self.K-len(esclusi)) for i,e in enumerate(vec)]
        return vec

    def discussion(self,x,uke):
        p_agree=self.p_agreement(x,uke)
        l=np.random.random_integers(0,self.K-1)  #component modified by interaction
        coppia=np.asarray([self.v[x],uke])

        if np.random.rand()<p_agree: #interaction rule 
            agree=1.
        else:                   
            agree=-1.

        diff=coppia[1][l]-coppia[0][l]
        if np.abs(diff)<self.alpha:
            increment=0.5*(diff)
        else:
            increment=self.alpha*np.sign(diff)

        if (np.abs(increment)>coppia[0][l] and agree*increment<0) or (np.abs(increment)>(1-coppia[0][l]) and agree*increment>0):
            increment=np.amin([coppia[0][l], 1.-coppia[0][l]])

        coppia[0][l]=coppia[0][l]+agree*increment

        return self.renormalize(coppia[0], l, agree, increment)

    def interaction(self,x,y):
        self.v[x]=self.discussion(x,self.v[y])# first argument is a position, second is a vector    
        if np.random.rand()<self.p_I:    # the logic is that arg 1 must be an element of the pop, to be changed 
            if self.I.shape[0]==1:       # arg 2 is a generic vec that interatcs with the pop
                j=0
            else:
                j=np.random.choice(self.I.shape[0])
                self.v[x]=self.discussion(x,self.I[j])
        return self.v[x]

class unbiased(base):
    kind='unbiased'
    def __init__(self,N,K,S,e,alpha,p_I,P,m,n,*args):
        base.__init__(self,N,K,S,e,alpha,p_I,P,m)
        self.n=n
        self.personality=[1 if i<n*N else 0 for i in range(N)]
        
    def p_agreement(self,x,v2):
        if self.personality[x]==0:
            return np.clip(my.overlap(self.v[x],v2)+np.random.choice([self.e,-self.e]), 0, 1)
        else:
            return 0.5
        
class suggestible(base):
    kind='suggestible'
    def __init__(self,N,K,S,e,alpha,p_I,P,m,n,*args):
        unbiased.__init__(self,N,K,S,e,alpha,p_I,P,m,n)
        
    def p_agreement(self,x,v2):
        if self.personality[x]==0:
            return np.clip(my.overlap(self.v[x],v2)+np.random.choice([self.e,-self.e]), 0, 1)
        else:
            return 1.        
    
class fixedp(unbiased):
    kind='fixedp'
    def __init__(self,N,K,S,e,alpha,p_I,P,m,n,pa,*args):
        unbiased.__init__(self,N,K,S,e,alpha,p_I,P,m,n)
        self.pa=pa

    def p_agreement(self,x,v2):
        if self.personality[x]==0:
            return np.clip(my.overlap(self.v[x],v2)+np.random.choice([self.e,-self.e]), 0, 1)
        else:
            return self.pa


