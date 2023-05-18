import numpy as np
import pylab as plt
import scipy.linalg as sl

# Function to obtain the maximal perturbation to preserve the extremal value of Fiedler vector

def perturbed_laplacian(L,node,t):
    n=L.shape[0]
    Lperturbed=np.zeros((n+1,n+1))
    Lperturbed[0:n,0:n]=L
    Lperturbed[node,node]=L[node,node]+t
    Lperturbed[node,n]=-t
    Lperturbed[n,node]=-t
    Lperturbed[n,n]=t
    return Lperturbed

def maximal_perturbation(L,node,t_sampled):
    n=L.shape[0]
    Lperturbed=np.zeros((n+1,n+1))
    Lperturbed[0:n,0:n]=L
    all_val=np.zeros((n+1,len(t_sampled)))
    #all_vec=np.zeros((n+1,n+1,len(t_sampled)))
    all_vec=np.zeros((n+1,2,len(t_sampled)))
    for i,t in enumerate(t_sampled):
        Lperturbed[node,node]=L[node,node]+t
        Lperturbed[node,n]=-t
        Lperturbed[n,node]=-t
        Lperturbed[n,n]=t
        #
        try:
            val, vec=sl.eigh(Lperturbed,subset_by_index=[0,1])
        except:
            val,vec=np.linalg.eig(Lperturbed)
            val = val[np.argsort(val)]
            vec = vec[:,np.argsort(val)]
            vec = vec[:,0:2]
            print("Error with eigh. Full solver is used.")
        all_vec[:,:,i] = vec
    return (all_val,all_vec)

def line_graph(n):
	A = np.zeros((n,n))
	for i in range(n-1):
		A[i,i+1] = 1
		A[i+1,i] = 1
	L=np.diag(A.sum(axis=0))-A
	return L
	
def complete_graph(n):
	A = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			if i!=j:
				A[i,j] = 1
	L=np.diag(A.sum(axis=0))-A
	return L

def threshold_centrality(L,t_sampled=np.arange(1e-10,2,1e-3)):
    n = len(L)
    max_perturbation=np.zeros(n)
    lim_algebraic_connectivity=np.zeros(n)
    all_val_all_nodes=np.zeros((n,len(t_sampled)))
    threshold_t = np.zeros((n,))-1
    for node in range(n):
        all_val,all_vec=maximal_perturbation(L,node,t_sampled)
        all_fiedler=np.sign(all_vec[n,1,:])*all_vec[:,1,:] # 1 for vp 1 = Fiedler
        t=(all_fiedler[n,:]<all_fiedler.max(axis=0)).nonzero()[0] # it is: 
        #print(t)
        # n: where it is perturbed
        if t.size>0:
            max_perturbation[node]=t_sampled[t[0]]
            threshold_t[node] = t_sampled[t[0]]
        else:
            threshold_t[node] = np.inf
        all_val_all_nodes[node,:]=all_val[1,:]
        lim_algebraic_connectivity[node]=all_val[1,len(t_sampled)-1]
    return threshold_t
