import numpy as np
from numpy.linalg import norm
import time

def cosine(c,m): #c = centroids m = stems matrix
    centroidi = {el:[] for el in range(m.shape[0])}
    for i in range(m.shape[0]):
        for j in range(c.shape[0]):
            centroidi[i].append(np.dot(m[i],c[j])/(norm(m[i])*norm(c[j])))
    return(centroidi)

def centroidi(m,g): #matrix, clusters dict
    centroide=np.zeros(shape=(len(g.keys()),m.shape[1]))
    for i in range(len(g.keys())):
        centroide[i]=np.sum(m[g[i]], axis=0)/len(g[i])
    return(centroide)

def kmeans(m,c): #m = stems matrix, c = centroids
    
# With the following command, you obtain a dictionary where the keys are the row indices of each book, and the values are lists containing the distances of the book from each of the 2 centroids.
    print("running...")
    k=c.shape[0]
    distances=cosine(c,m)
# With the following command, to each list of distances, the index of the centroid that has the highest similarity with that book is added.
    for z in range(len(distances.keys())):
        distances[z]=[distances[z],[[i for i, j in enumerate(distances[z]) if j == max(distances[z])][0]]]
    groups={el:[] for el in range(k)}
    centroids=list(range(k))
    for i in centroids:
        for j in distances.keys():
            if distances[j][1][0]==centroids[i]: groups[i].append(j)
    cohesion=k*[0]
    for i in range(len(distances.keys())):
        cohesion[distances[i][1][0]]+=distances[i][0][distances[i][1][0]]
    cohesion.append([sum(cohesion)]) # total coesion
    
    # new centroids:
    centroids_new=centroidi(m, groups)
    distances_new=cosine(centroids_new,m)
    for z in range(len(distances_new.keys())):
        distances_new[z]=[distances_new[z],[[i for i, j in enumerate(distances_new[z]) if j == max(distances_new[z])][0]]]
    groups_new={el:[] for el in range(k)}
    centroids=list(range(k))
    for i in centroids:
        for j in distances_new.keys():
            if distances_new[j][1][0]==centroids[i]: groups_new[i].append(j) 
    cohesion_new=k*[0]
    for i in range(len(distances_new.keys())):
        cohesion_new[distances_new[i][1][0]]+=distances_new[i][0][distances_new[i][1][0]]
    cohesion_new.append([sum(cohesion_new)])
    
    b=1
    print("iteration number {}".format(b))

    start_time = time.time()
    b=1    
    while True:
        groups=groups_new
        k_centroids=centroids_new
        cohesion=cohesion_new
        centroids_new=centroidi(m,groups)
        distances_new=cosine(centroids_new,m)
        for z in range(len(distances_new.keys())):
            distances_new[z]=[distances_new[z],[[i for i, j in enumerate(distances_new[z]) if j == max(distances_new[z])][0]]]
    
        groups_new={el:[] for el in range(k)}
        centroids=list(range(k))
        for i in centroids:
            for j in distances_new.keys():
                if distances_new[j][1][0]==centroids[i]: groups_new[i].append(j) 
    
        
        cohesion_new=k*[0]
        for i in range(len(distances_new.keys())):
            cohesion_new[distances_new[i][1][0]]+=distances_new[i][0][distances_new[i][1][0]]
        cohesion_new.append([sum(cohesion_new)]) 
        if cohesion_new[-1][0]/cohesion[-1][0]<=1:break
        else:b=b+1
        print("iteration number {}".format(b))
    
    print("time spent: %s seconds" %(time.time() - start_time))
    print("number of iteration: {}".format(b))
    return(groups)

def spectral_cl(m, k): # m = matrix, k = num clusters
    p = m.shape[1]
    s = m.shape[0]
    W = np.empty((s, s))
    for i, j in np.ndindex(W.shape):
        W[i, j] = cosine(m[i,].reshape(1, -1), m[j,].reshape(1, -1))[0][0]
    D = np.diag(np.sum(W, axis = 0))
    L = D - W
    # D^(1/2) * L * D^(1/2)
    L_sym = np.dot(np.dot(np.diag(1/np.sqrt(np.diag(D))), L), np.diag(1/np.sqrt(np.diag(D))))
    e_val, e_vec = np.linalg.eig(L_sym)
    T = U = e_vec[:, :k]
    cl = kmeans(T,T[[0, 1]])
    return cl
