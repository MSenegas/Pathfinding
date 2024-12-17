import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class DirMap:
    def __init__(self,dim_U):
        self.D=-2*np.ones(dim_U,dtype=np.int8)
        self.i_A=-1
        self.j_A=-1
    def dir_U(self,U,i_A,j_A,i_D0,j_D0,GU=None):
        '''Remplit la matrice des directions pour la destination A'''
        dim_U=np.shape(U)
        if max(dim_U)<=32:
            if i_A!=self.i_A or j_A!=self.j_A:
                for i in range(dim_U[0]):
                    for j in range(dim_U[1]):
                        self.D[i,j]=succ(U,i,j,i_A,j_A)
                        self.i_A=i_A
                        self.j_A=j_A
        else:
            if i_A!=self.i_A or j_A!=self.j_A:
                self.D=-2*np.ones(np.shape(U),dtype=np.int8)
                self.i_A=i_A
                self.j_A=j_A
            i_G=i_D0//32;j_G=j_D0//32
            i_T=i_A//32;j_T=j_A//32
            if self.D[32*i_G,32*j_G]==-2:
                if GU is None:
                    GU=clustering(U)
                prd_G=scipy.sparse.csgraph.floyd_warshall(W_U(GU),return_predecessors=True)[1]
                LU=U[i_G*32:i_G*32+32,j_G*32:j_G*32+32]
                dist_l,prd_l=scipy.sparse.csgraph.floyd_warshall(W_U(LU),return_predecessors=True)
                d_G=succ(GU,i_G,j_G,i_T,j_T,prd_G)
                for i in range(32*i_G,32*i_G+32):
                    i_l=i%32
                    for j in range(32*j_G,32*j_G+32):
                        j_l=j%32
                        i_B,j_B=local_target(U,i,j,i_A,j_A,GU,prd_G,dist_l,d_G)
                        d_l=succ(LU,i_l,j_l,i_B,j_B,prd_l)
                        if d_l==-1:
                            self.D[i,j]=d_G
                        else:
                            self.D[i,j]=d_l

def W_U(U):
    '''Transforme la matrice de poids U en un format utilisable par floyd_warshall'''
    assert (U>0).all(), 'Les poids doivent être strictement positifs'
    rep=scipy.sparse.lil_matrix((np.size(U),np.size(U)))
    dim_U=np.shape(U)
    for i in range(dim_U[0]):
        for j in range(dim_U[1]):
            if i!=0:
                rep[dim_U[1]*i+j,dim_U[1]*(i-1)+j]=U[i-1,j]
            if j!=0:
                rep[dim_U[1]*i+j,dim_U[1]*i+j-1]=U[i,j-1]
            if j!=dim_U[0]-1:
                rep[dim_U[1]*i+j,dim_U[1]*i+j+1]=U[i,j+1]
            if i!=dim_U[0]-1:
                rep[dim_U[1]*i+j,dim_U[1]*(i+1)+j]=U[i+1,j]
    return scipy.sparse.csr_matrix(rep)

# prd=scipy.sparse.csgraph.floyd_warshall(W_U(U),return_predecessors=True)[1]

def succ(U,i_D,j_D,i_A,j_A,prd=None):
    '''Calcule la direction pour aller de D à A'''
    dim_U=np.shape(U)
    if prd is None:
        prd=scipy.sparse.csgraph.floyd_warshall(W_U(U),return_predecessors=True)[1]
    jj=dim_U[1]*i_A+j_A
    ii=dim_U[1]*i_D+j_D
    # Hyp parcourir chemin dans les deux sens idem
    kk=prd[jj,ii]
    if kk!=-9999:
        i_N=kk//dim_U[1]
        j_N=kk%dim_U[1]
    else:
        return -1 # Inaccessible ou déjà arrivé
    if j_N==j_D+1:
        return 0 # Est
    elif j_N==j_D-1:
        return 2 # Ouest
    elif i_N==i_D+1:
        return 1 # Sud
    elif i_N==i_D-1:
        return 3 # Nord

def clustering(U):
    '''Crée des clusters de taille 32*32 pour réduire le temps et mémoire nécessaires au calcul
    Renvoie la matrice des poids moyens des clusters'''
    dim_U=np.shape(U)
    GU=np.zeros((dim_U[0]//32,dim_U[1]//32))
    for i in range(np.shape(GU)[0]):
        for j in range(np.shape(GU)[1]):
            GU[i,j]=np.mean(U[i*32:i*32+32,j*32:j*32+32])
    return GU

def local_target(U,i_D,j_D,i_A,j_A,GU=None,prd_G=None,dist_l=None,d=None):
    '''Détermine l'équivalent local B de la destination globale A'''
    dim_U=np.shape(U)
    assert dim_U[0]%32+dim_U[1]%32==0, 'Côtés de la carte doit être un multiple de 32'
    i_l=i_D%32;j_l=j_D%32
    i_G=i_D//32;j_G=j_D//32
    i_T=i_A//32;j_T=j_A//32
    LU=U[i_G*32:i_G*32+32,j_G*32:j_G*32+32]
    if 32*i_G<=i_A and i_A<32*i_G+32 and 32*j_G<=j_A and j_A<32*j_G+32:
        return i_A%32;j_A%32
    if GU is None:
        GU=clustering(U)
    if dist_l is None:
        dist_l=scipy.sparse.csgraph.floyd_warshall(W_U(LU))
    if d is None:
        d=succ(GU,i_G,j_G,i_T,j_T,prd_G)
    if d==-1:
        return i_l,j_l
    dist_b=dist_l[32*i_l+j_l]
    if d==0:
        kk=np.argmin([dist_b[32*i-1] for i in range(1,33)])
        i_B=kk%32
        j_B=32-1
    elif d==1:
        kk=np.argmin([dist_b[j-32] for j in range(32)])
        j_B=kk%32
        i_B=32-1
    elif d==2:
        kk=np.argmin([dist_b[32*i] for i in range(32)])
        i_B=kk%32
        j_B=0
    elif d==3:
        kk=np.argmin([dist_b[j] for j in range(32)])
        j_B=kk%32
        i_B=0
    return i_B,j_B