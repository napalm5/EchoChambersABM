import argparse
import itertools as it
import sys
import importlib
import numpy as np
import scipy as sp
from scipy import stats
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.tri as tri
#import gc

def generate(N,K):
    v=np.random.rand(N,K-1)
    v=np.apply_along_axis(np.append, 1, v, [0,1])
    v=np.apply_along_axis(np.sort, 1, v)
    v=np.apply_along_axis(np.ediff1d, 1, v)
    return v

def good_agent(K):
    agente=np.random.rand(K-1)
    agente=np.append(agente,[0,1])
    agente=np.sort(agente)
    agente=np.ediff1d(agente)
    return agente

def generate_with_selection(N,K,S):
# discard individuals with Entropy over S
# Probably could be done in a more efficient way, like the other function
    pop=np.zeros([N,K])
    for i in range(0,N):
        agente=good_agent(K)
        while stats.entropy(agente)>S:
            agente=good_agent(K)            
        pop[i]=agente
    return pop

def information(K,P,m):
    Q=(1-P)/(K-1)
    if m:                        #external information
        I=[[P if i is j else Q for i in range(K)] for j in range(K)]
        I=np.asarray(I,float)   
    else:
        I=np.ndarray([1,K])        
        I[0]=[Q for i in range(K)]
        I[0,0]=P
    return I
        
def overlap(a1,a2):
#    o = np.dot(a1,a2)/(np.linalg.norm(a1)*np.linalg.norm(a2))
    return (1-sp.spatial.distance.cosine(a1,a2))

def cohesion(pop,N):
    return 2*sum(overlap(x,y) for x, y in it.combinations(pop, 2))/(N*(N-1))

def information_overlap(pop,I): #TODO: remove the N argument
    return sum(overlap(x,I) for x in pop)/pop.shape[0]

def s_io(pop,I):
    overlaps=[information_overlap(pop,i) for i in I]
#    print(I)
#    print(overlaps)
    overlaps=overlaps/np.linalg.norm(overlaps)
    return stats.entropy(overlaps)

def dispersion(v,freq,clusters,K):
    N=v.shape[0]
    if freq.size >= K and freq.size<K*2. and (np.sort(freq)[-K:]>max(N/(3*K),5)).sum()==K:  #i.e. if the clusters are roughly K and each cluster has at least N/(K*2) members
        #    cl=[[a for a in v if my.ispolarized(a) and my.vote(a)==i] for i in [1,2,3]]
        cl=[[v[j] for j in range(clusters.size) if clusters[j]==i] for i in np.unique(clusters)[-K:]]
        av=[std(np.asarray(m)) for m in cl] #insert here your measure of the dispersion of the clusters
        print(av)
        disp=np.average(av)
        print('Done')
    else:
        disp='nan'
        print('The population of the clusters is:')
        print(freq)
        print('No significative clusters')
    return disp

def average_entropy(pop):
    en=stats.entropy(list(map(list, zip(*pop)))) #dunno if it's a good way to do it
    #also: [stats.entropy(v) for v in pop]
    return np.mean(en)

def renormalize(vec,K,l,agree,increment):
    # # renormalization (the lazy way)
    # Problem with this: zero values remain zero
    # norm = 1/sum(coppia[i])
    # coppia[i]=coppia[i]*norm

    # #proper renormalization
    vec=[e if i==l else e-agree*increment/(K-1) for i,e in enumerate(vec)]
    esclusi=[l]
    while np.amin(vec)<0:
        increment=np.abs(np.amin(vec))
        minimum=np.argmin(vec)
        esclusi.append(minimum)
        vec[minimum]=0
        vec=[e if i in esclusi else e-increment/(K-len(esclusi)) for i,e in enumerate(vec)]

    return vec
    
def discussion(a1,a2,p1,p2,K,e,alpha,pa):
    if p1==0:
        p_agree=np.clip(overlap(a1,a2)+np.random.choice([e,-e]), 0, 1)
        p_disagree=1-p_agree
    else:
        p_agree=pa
        p_disagree=(1-pa)
        
    l=np.random.random_integers(0,K-1)
    
    coppia=[a1,a2]
       
    if np.random.rand()<p_agree: #interaction rule 
        agree=1.
    else:                   
        agree=-1.
        
    if np.abs(a1[l]-a2[l])<alpha:
        increment=0.5*(coppia[1][l]-coppia[0][l])
    else:
        increment=alpha*np.sign(coppia[1][l]-coppia[0][l])

    if (np.abs(increment)>coppia[0][l] and agree*increment<0) or (np.abs(increment)>(1-coppia[0][l]) and agree*increment>0):
        increment=min(coppia[0][l], 1-coppia[0][l])
        
    coppia[0][l]=coppia[0][l]+agree*increment

    coppia[0]=renormalize(coppia[0], K, l, agree, increment)
    return coppia

    
def interaction(a1,a2,p1,p2,K,e,alpha,I,p_I,pa):

#    i=np.random.choice([0,1])
#    i=0
    coppia=discussion(a1,a2,p1,p2,K,e,alpha,pa)    
#    print('---before media')
#    print(coppia[i])
    if np.random.rand()<p_I:
#        print('MEDIA!')
        if I.shape[0]==1:
            n=0
        else:
            n=np.random.choice(I.shape[0])
            coppia[0]=discussion(coppia[0],I[n],p1,0,K,e,alpha,pa)[0] 

#    print('---end interaction---')
    return (coppia[0],coppia[1])

def PR(c):
    return np.sum(c)**2/np.sum(c**2)


def plot(v):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for a in v:
        xs=a[0]
        ys=a[1]
        zs=a[2]
        ax.scatter(xs, ys, zs, c='b')    

    ax.set_xlabel('Opinione X')
    ax.set_ylabel('Opinione Y')
    ax.set_zlabel('Opinione Z')

    plt.show(block=True)
    return fig

def colorplot(v,c, file=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    maxcol=np.amax(c)
    col=plt.cm.rainbow(c/maxcol)

    ax.scatter(v[:,0],v[:,1],v[:,2], c=col[:])
#    for a,co in zip(v,col):
#        ax.scatter(a[0],a[1],a[2], c=co)
    
    ax.set_xlabel('Opinione X')
    ax.set_ylabel('Opinione Y')
    ax.set_zlabel('Opinione Z')

    ax.view_init(azim=30)
    
    if file is None:  #commented for LCM condor compatibility
#        plt.show(block=True)  #(there probably is a better way)
        return fig
    else:
        plt.savefig(file)
    
    return fig

def specialplot(v,col, file=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(v[:,0],v[:,1],v[:,2], c=col[:])
    
    ax.set_xlabel('Opinione X')
    ax.set_ylabel('Opinione Y')
    ax.set_zlabel('Opinione Z')

    ax.view_init(azim=35)
    
    if file is None:  #commented for LCM condor compatibility
#        plt.show(block=True)  #(there probably is a better way)
        return fig
    else:
        plt.savefig(file)

    gc.collect()
    return fig

def triangleplot(v):
    #corners
    x=np.array([0,1,0.5])
    y=np.array([0,0,np.sqrt(3)*0.5])

    # create a triangulation out of these points
    T = tri.Triangulation(x,y)

    # plot the contour
    #plt.tricontourf(x,y,T.triangles,v)


    # create the grid
    corners = np.array([[0, 0], [1, 0], [0.5,  np.sqrt(3)*0.5]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=4)

    #plotting the mesh
    plt.triplot(trimesh,'-', color=[0.9,0.9,0.9])
    plt.triplot(T, color=[0.2,0.2,0.2])

#    v=[[1,0,0],[0,1,0]]
    c0=np.array([0.,0.])
    c1=np.array([1.,0.])
    c2=np.array([0.5,np.sqrt(3)*0.5])
    data=np.asarray([c0*i[0]+c1*i[1]+c2*i[2] for i in v])
    #Plotting the data
    plt.scatter(data[:,0], data[:,1], s=10, zorder = 10)#, cmap=colormap)
    plt.gca().annotate('a',xy=c0,xytext=(c0+[-0.03,-0.03]))
    plt.gca().annotate('b',xy=c1,xytext=(c1+[+0.02,-0.03]))
    plt.gca().annotate('c',xy=c2,xytext=(c2+[-0.01,+0.03]))
        
    plt.axis('off')
    plt.show()
    gc.collect()
    return

def centroid(vec):
    return  np.asarray([np.mean(col) for col in vec.T])

def std(vec):
    center=centroid(vec)
    dist=np.asarray([sp.spatial.distance.euclidean(v,center) for v in vec])
    return np.sqrt(np.sum(dist**2))


#---Mobility measurement---

def ispolarized(agent):
#    cutoff=0.28 #obtained from previous graphs
    # cutoff=0.65
    # if stats.entropy(agent)>cutoff:
    #     return 0
    cutoff=0.45
    if sp.spatial.distance.euclidean(agent,np.round(agent)) < cutoff:
        return 1
    else:
        return 0
def vote(agent):
    return np.argmax(agent)+1

polarization=np.vectorize(lambda a: 0 if stats.entropy(a)>0.28 else 1, signature='(m)->()')
votes=np.vectorize(lambda agent: np.argmax(agent)+1, signature='(m)->()')


#--- simplex stuff

#Once you've chosen your vertices (say p1,…,p4), the mapping is just (x1,…,x4)↦x1p1+⋯+x4p4.
def cart_to_bar(p):
    return 0

def tofile(output,start,stop,universe,K,N,S,e,alpha,p_I,t,o_t,P,n,m,initial_cohesion,initial_IO,final_dispersion,final_IO,PR,SIO,M1,M2,Si_norm):
    with open(output, 'w') as f:
        value=str('***This program ran in: '+str(stop - start)+'s***')
        value=value+'\n'+'\n'
        f.write(value)

        #print parameters
        f.write('#'+universe.kind+'\n')
        f.write('#K = '+str(K)+'\n')
        f.write('#N = '+str(N)+'\n')
        f.write('#S = '+str(S)+'\n')
        f.write('#e = '+str(e)+'\n')
        f.write('#alpha = '+str(alpha)+'\n')
        f.write('#p_I = '+str(p_I)+'\n')
        f.write('#t = '+str(t)+'\n')
        f.write('#o_t = '+str(o_t)+'\n')
        f.write('#P = '+str(P)+'\n')
        f.write('#n = '+str(n)+'\n')
        f.write('#m = '+str(m)+'\n')    
        f.write('\n')
    
        #print results
        value=str('K InitialCohesion InitialIO FinalDispersion FinalIO Si_norm PR S_IO M1 M2\n')
        f.write(value)
        value=str(K)+'     '+initial_cohesion+'     '+initial_IO+'     '+final_dispersion+'     '+final_IO+'     '+Si_norm+'     '+PR+'     '+SIO+'     '+M1+'     '+M2+'\n'
        f.write(value)
    

def check_args(args):
    parser=argparse.ArgumentParser(description='Opinion dynamics simulator')
    parser.add_argument('kind', type=str, help='Flavor of the simulation')
    parser.add_argument('-K', '--choices', type=int, help='Number of choices', required='True' )
    parser.add_argument('-N', '--agents', type=int, help='Number of agents', required='True')
    parser.add_argument('-S', '--entropy', type=float, help='Entropy relative cutoff (to be substituted by initial overlap)', default=1.)
    parser.add_argument('-pI', '--mediainfluence', type=float, help='Probability of interaction with external media', default=0.)
    parser.add_argument('-t', '--steps', type=int, help='Number of steps of the simulation', required='True')
    parser.add_argument('-ot', '--threshold', type=float, help='Clustering threshold', default=0.8 )
    parser.add_argument('-P', '--mediapolarization', type=float, help='Polarization of media information', default=0.8)
    parser.add_argument('-a', '--alpha', type=float, help='Flexibility of the agents', default=0.01)
    parser.add_argument('-n', '--special', type=float, help='Fraction of special agents', default=0.)
    parser.add_argument('-pa', '--pagree', type=float, help='Probability of agreement for unbiased agents', default=0.5)
    parser.add_argument('-m', '--multiple', action='store_true', help='Use multiple information sources')
    parser.add_argument('-f', '--file', help='where to write the output')
    parser.add_argument('-g', '--graph', help='where to print the graph')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    result=parser.parse_args()
    
    #Argument dependencies
#    if result.mediainfluence is not None and result.mediapolarization is None:
 #       parser.error("Option -pI requires the setting -P")
    
    return result
