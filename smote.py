import math
import random
from operator import itemgetter
import dataset
import pdb

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'                                   SMOTEing (Chawla, et. al.)                                         '
'                   We generate synthetic minority class examples to balance the training set.         '
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def SMOTE(points,N,k=5):
    synthetics = []
    # Assuming N%100 = 0
    print "SMOTING %s percent." % N
    N = N/100
    for p in points:
        nnarray = k_nearest_neighbors(p, points, k)
        synthetics.extend(populate(N, p, nnarray))
    return synthetics
    

def SMOTE_n_points(points, n, k=5):
    synthetics = []
    # Assuming N%100 = 0
   # print "SMOTEing %s synthetics." % n
    while len(synthetics) < n:
        p = random.choice(points)
        nnarray = k_nearest_neighbors(p, points, k)
        synthetics.extend(populate(1, p, nnarray))
    return synthetics
    
def dict_to_ls_point(d_points):
    key_vals = []
    for dp in d_points:
        key_vals.extend(dp.keys())
    ls_points = []
    for dp in d_points:
        cur_ls_point = []
        for v in range(max(key_vals)):
            if not dp.has_key(v+1):
                cur_ls_point.append(0.0)
            else:
                cur_ls_point.append(dp[v+1])
        ls_points.append(cur_ls_point)
    return ls_points

	
def k_nearest_neighbors(p, neighbors, k):
    '''
    finds k nearest neighbors to p, picks one at random and returns it.
    
    neighbors needs to be a collection of instance objects.
    '''
    neighbors_to_distances = {}
    
    for neighbor in neighbors:
        if neighbor.id != p.id:
            neighbors_to_distances[neighbor.id] = euclid_dist(p.point, neighbor.point)
    
    ids_sorted_by_dist = sorted(neighbors_to_distances.items(), key=itemgetter(1))
    closest_ids = [id_dist_pair[0] for id_dist_pair in ids_sorted_by_dist[:k]]
    return [x.point for x in [neighbor for neighbor in neighbors if neighbor.id in closest_ids]]
 

def euclid_dist(p1, p2):
    distance = 0.0
    shared_coords = intersection(p1.keys(), p2.keys()) 
    for coord in shared_coords:
        distance += math.pow(abs(p1[coord]-p2[coord]), 2)
    for coord in [x for x in p1.keys() if not x in shared_coords]:
        distance += math.pow(abs(p1[coord]), 2)
    for coord in [x for x in p2.keys() if not x in shared_coords]:
        distance += math.pow(abs(p2[coord]), 2)
            
    return math.sqrt(distance)

def intersection(set_1, set_2):
    return [s for s in set_1 if s in set_2]

    
def populate(N, inst, nnarray):
    synthetics = []
    p = inst.point
    while N > 0:
        neighbor = random.choice(nnarray)
        synthetic = {}
        shared_coords = intersection(p.keys(), neighbor.keys())
        gap = random.random()
        for coord in shared_coords:
            dif = abs(p[coord] - neighbor[coord])
            synthetic[coord] = neighbor[coord] + gap*dif
        for coord in [coord for coord in p.keys() if not coord in neighbor.keys()]:
            dif = abs(p[coord])
            synthetic[coord] = 0.0 + gap*dif
        for coord in [coord for coord in neighbor.keys() if not coord in p.keys()]:
            dif = abs(neighbor[coord])
            synthetic[coord] = neighbor[coord] + gap*dif
            
        N = N-1
        synthetics.append(dataset.instance(-1, synthetic, label=1, is_synthetic=True)) 
    return synthetics
        

def euclidDist(a,b):
    '''
    Computes and returns the Euclidean distance between points a and b in n-dimensional space.
    '''
    if len(a) != len(b):
        print "points must be the same length!"
        return None
    else:
        total = 0
        for d in range(len(a)):
            total+=math.pow(abs(a[d]-b[d]), 2)
        return math.sqrt(total)
        
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'                                   END SMOTEing.                                                      '
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''