import heapq
import numpy as np

def A_star(U,i_D,j_D,i_A,j_A):
    '''https://fr.wikipedia.org/wiki/Algorithme_A*'''
    prd=[]
    wrk=[]
    # Format (h,c,i,j,ky)
    # h: heuristique, c: coût
    # i: ligne, j: colonne
    # ky: indice du prédécesseur
    heapq.heappush(wrk,(0,0,i_D,j_D,-1))
    while len(wrk)>0:
        u=heapq.heappop(wrk)
        if u[2]==i_A and u[3]==j_A:
            return build_path(prd,u,U)
        ngb=neighbors(U,u,i_A,j_A,len(prd))
        for v in ngb:
            if cond_new(prd,wrk,v):
                heapq.heappush(wrk, v)
        prd.append(u)

def dist(i,j,i_A,j_A):
    '''Renvoie la distance en norme 1'''
    return abs(i_A-i)+abs(j_A-j)

def f_heur(i,j,i_A,j_A,min_U):
    '''Évalue la fonction à ajouter au coût pour obtenir l'heuristique'''
    return dist(i,j,i_A,j_A)*min_U*2

def build_path(prd,u,U):
    '''Reconstitue le chemin'''
    rep=[]
    v=u
    while v[4]!=-1:
        rep.append((v[2],v[3]))
        v=prd[v[4]]
    rep.append((v[2],v[3]))
    rep.reverse()
    return rep

def move(i,j,d):
    '''Transforme un position et une direction en un voisin'''
    if d==0:
        return i,j+1 # Est
    if d==1:
        return i+1,j # Sud
    if d==2:
        return i,j-1 # Ouest
    if d==3:
        return i-1,j # Nord

def direction(D,v):
    '''Renvoie la direction pour aller de D=i,j à un voisin v'''
    (i,j),(i_v,j_v)=D,v
    assert dist(i,j,i_v,j_v)==1, 'Ceci n\'est pas un voisin'
    if j_v==j+1:
        return 0 # Est
    if i_v==i+1:
        return 1 # Sud
    if j_v==j-1:
        return 2 # Ouest
    if i_v==i-1:
        return 3 # Nord

def neighbors(U,u,i_A,j_A,ky):
    '''Détermine les voisins avec leur coût dans l'algorithme A*'''
    dim_U=np.shape(U)
    ngb=[]
    for d in range(4):
        i,j=move(u[2],u[3],d)
        if 0<=i<dim_U[0] and 0<=j<dim_U[1] and U[i,j]<np.inf:
            ngb.append((f_heur(i,j,i_A,j_A,np.min(U))+u[1]+U[i,j],u[1]+U[i,j],i,j,ky))
    return ngb

def cond_new(prd,wrk,v):
    '''Détermine si le point v doit être ajouté à la file d'attente prioritaire'''
    h,c,i,j,ky=v
    if (i,j) in [(p[2],p[3]) for p in prd]:
        return False
    if not (i,j) in [(p[2],p[3]) for p in wrk]:
        return True
    n=[(p[2],p[3]) for p in wrk].index((i,j))
    if wrk[n][1]>c:
        return True
    return False

def clustering(U,cluster_shape):
    '''Découpe la carte en clusters'''
    if type(cluster_shape) is int:
        cluster_shape = (cluster_shape,cluster_shape)
    dim_U=np.shape(U)
    assert (np.array(dim_U)%np.array(cluster_shape)==0).all(), 'Dimensions des clusters incompatibles avec la carte'
    dim_G=tuple(np.array(dim_U)//np.array(cluster_shape))
    GU=np.ones(dim_G)
    for i in range(dim_G[0]):
        for j in range(dim_G[1]):
            GU[i,j]=np.mean(U[cluster_shape[0]*i:cluster_shape[0]*(i+1),cluster_shape[1]*j:cluster_shape[1]*(j+1)])
    return GU

def local_target(U,chemin_G,cluster_shape,i_D,j_D):
    '''Trouve l'équivalent local de la destination globale sur le cluster voisin'''
    d=direction(chemin_G[0],chemin_G[1])
    if len(chemin_G)==2:
        d_=-1
    else:
        d_=direction(chemin_G[1],chemin_G[2])
    i_a,j_a=i_D,j_D
    i_G,j_G=np.array((i_D,j_D))//np.array(cluster_shape)
    if 0 in (d,d_):
        i_a=(i_G+1)*cluster_shape[0]-1
    elif 2 in (d,d_):
        i_a=i_G*cluster_shape[0]
    if 1 in (d,d_):
        j_a=(j_G+1)*cluster_shape[1]-1  
    elif 3 in (d,d_):
        j_a=j_G*cluster_shape[1]
    return i_a,j_a

def local_path(U,chemin_G,cluster_shape,i_D,j_D,i_A,j_A):
    '''Renvoie le chemin à l'échelle locale'''
    i_G,j_G=np.array((i_D,j_D))//np.array(cluster_shape)
    i_l,j_l=np.array((i_D,j_D))%np.array(cluster_shape)
    LU=U[cluster_shape[0]*i_G:cluster_shape[0]*(i_G+1),cluster_shape[1]*j_G:cluster_shape[1]*(j_G+1)]
    if len(chemin_G)==1:
        i_la,j_la=np.array((i_A,j_A))%np.array(cluster_shape)
    else:
        i_la,j_la=np.array(local_target(U,chemin_G,cluster_shape,i_D,j_D))%np.array(cluster_shape)
    ll=np.array(A_star(LU,i_l,j_l,i_la,j_la))
    ll+=np.array((i_G,j_G))*np.array(cluster_shape)
    ll=[tuple(t) for t in ll]
    if len(chemin_G)>1:
        ll.append(move(ll[-1][0],ll[-1][1],direction(chemin_G[0],chemin_G[1])))
    return ll

def cluster_paths(U,GU,i_D,j_D,i_A,j_A):
    '''Renvoie les chemins aux deux échelles'''
    dim_U=np.shape(U)
    dim_G=np.shape(GU)
    cluster_shape=tuple(np.array(dim_U)//np.array(dim_G))
    i_G,j_G=np.array((i_D,j_D))//np.array(cluster_shape)
    i_T,j_T=np.array((i_A,j_A))//np.array(cluster_shape)
    chemin_G=A_star(U,i_G,j_G,i_T,j_T)
    chemin=local_path(U,chemin_G,cluster_shape,i_D,j_D,i_A,j_A)
    return chemin,chemin_G