#!/usr/bin/env python3

import argparse
import itertools as it
import sys
import importlib
import numpy as np
import scipy as sp
from scipy import stats
import scipy.cluster.hierarchy as hac
#import matplotlib
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#matplotlib.use('Agg')
#import gc
import timeit

#mymodules
from echocABM import functions as my
from echocABM import model as md

#numpy style settings
#np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True) 

#TimeIt!
start = timeit.default_timer()

par=my.check_args(sys.argv[1:])

output=par.file
graph=par.graph
decimals=4

#Parameters of the model

K=par.choices                # number of choices                    [2,+inf)
N=par.agents                 # number of agents                     [0,+inf)
S=par.entropy                # entropy (relative) cutoff            [0,1]
e=0.1                        # epsilon of interaction               [0,1]
alpha=par.alpha              # cutoff of interaction                [0,1]
p_I=par.mediainfluence       # influence of external media          [0,1]
t=par.steps                  # interaction time                     [0,+inf)
o_t=par.threshold            # clustering threshold                 [0,1] 
P=par.mediapolarization      # polarization of external information [0,1]
Q=(1-P)/(K-1)
m=par.multiple

kind=par.kind
n=par.special                       # fraction of special agents           [0,1]
pa=par.pagree

#The previous assignation is not really functional, but helps me to organize ideas

universe=md.base.factory(kind,*[N,K,S,e,alpha,p_I,P,m,n,pa])
#universe=md.model.factory(kind,*para.values()) #alternative approach
#universe=md.model.factory(kind,par) #another alternative approach
print('Using the '+universe.kind+' model')
v_0=universe.v

personality=np.zeros(N)

print('State of the universe at the beginning')
print(universe.personality)
print(universe.I)
print(universe.v)

initial_cohesion=str(round(my.cohesion(universe.v,universe.N), decimals))
initial_IO=str(round(my.information_overlap(universe.v,universe.I[0]), decimals))
print('The initial Cohesion is: o =',initial_cohesion)
print('The initial Information Overlap is: IO =',initial_IO)

# fig = plt.figure()
# plt.ion()
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(v[:,0],v[:,1],v[:,2], c='b')
# plt.draw()
#fig.show()
#fig.canvas.draw()

print('Socializing...')

#Experimental: track cluster occupation
state=np.zeros((N,2)) #s[i]=1 se v[i] e` in un angolo, 0 altrimenti
lastvote=np.zeros(N) #u[i]={1,2,3}, corrispondente all'ultimo (o al corrente) cluster in cui si e` trovato
mobility=np.zeros(N)
mobility2=np.zeros(N)
migrating=np.zeros(N)

for i in range(t):
    x=np.random.randint(N)
    y=np.random.randint(N)
    while y==x:
        y=np.random.randint(N)
    universe.interaction(x,y)

    # Mobility measurement
    if i==t/2:
        state[:,0]=list(map(my.ispolarized,universe.v))
        lastvote=list(map(my.vote,universe.v))
    if i>t/2:
        state[x,1]=my.ispolarized(universe.v[x])
        if state[x,0]>state[x,1]: #esco da un cluster
            mobility[x]+=1
            migrating[x]=1
        if state[x,0]<state[x,1]: #entro in un cluster
            migrating[x]=0
            if my.vote(universe.v[x])!=lastvote[x]: # se sono entrato in un altro cluster
                mobility2[x]+=1                 
                lastvote[x]=my.vote(universe.v[x]) 
        state[x,0]=state[x,1]

 
    # sc._offsets3d = (v[:,0],v[:,1],v[:,2])
    # if i%2000==0:
    #     plt.draw()
    #     plt.pause(0.001)

print('Completed in '+str(i)+' steps')

#From now on I will only consider normal agents
v=np.asarray([universe.v[i] for i in range(N) if universe.personality[i]==0])
mobility=np.asarray([mobility[i] for i in range(N) if universe.personality[i]==0])
mobility2=np.asarray([mobility2[i] for i in range(N) if universe.personality[i]==0])
print(mobility)
print(mobility2)
M1=str(np.average(mobility))
M2=str(np.average(mobility2))

np.savez("arrays."+str(K)+".N"+str(N)+".S"+str(par.entropy)+".pI"+str(p_I)+".ot"+str(o_t)+".P"+str(P)+".a"+str(alpha)+".n"+str(n)+".pa"+str(pa), v=v, m1=mobility, m2=mobility2, par=par, v_0=v_0)

#Here I could call analysis.py

print('the average mobility (1) is '+str(np.average(mobility)))
print('the average mobility (2) is '+str(np.average(mobility2)))

final_cohesion=str(round(my.cohesion(v,N), decimals))
final_IO=str(round(my.information_overlap(v,universe.I[0]), decimals))
final_entropy=str(my.average_entropy(v))
print('The final Cohesion is: o='+final_cohesion)
print('The final Information Overlap is: IO='+final_IO)

linkage=hac.linkage(v, metric=sp.spatial.distance.cosine, method='complete')
clusters=hac.fcluster(sp.clip(linkage,0,np.amax(linkage)), (1-o_t), criterion='distance')
freq=stats.itemfreq(clusters)[:,1]
pr=my.PR(freq)
PR=str(round(pr, decimals))
print('The Rartecipation Rate is: PR='+PR)
if m:
    SIO=str(round(my.s_io(v,universe.I),decimals))
else:
    SIO='nan'

dispersion=my.dispersion(v,freq,clusters,K)
final_dispersion=str(dispersion)

#if K==3:
#    my.triangleplot(v)
#    my.colorplot(v,clusters,graph)


stop = timeit.default_timer()
print('')
value=str('***This program ran in: '+str(stop - start)+'s***')
print (value) 


#----PRINT ON FILE-----
if output is not None:    
    my.tofile(output,start,stop,universe,universe.K,universe.N,universe.S,universe.e,universe.alpha,universe.p_I,t,o_t,universe.P,n,universe.m,initial_cohesion,initial_IO,final_dispersion,final_IO,PR,SIO,M1,M2,str(np.around(universe.S_I_norm,decimals)))
#this is error prone, should be rewritten with *args

