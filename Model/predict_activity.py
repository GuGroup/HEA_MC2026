import random
import math
from MCPredict import Predictor4MC
import json
import numpy as np
from ase.build import fcc111
from random import shuffle
from tqdm import tqdm
from ase.neighborlist import NeighborList


def GetOptimizedSlab(slab):
    initial_atomic_numbers = slab.get_atomic_numbers().tolist()
    s = slab.get_atomic_numbers()
    predictor = Predictor4MC(slab,'model_best.pth.tar','atom_init.json')

    kmax = 50000
    temperature_initial = 4000.0
    temperature_final = 298.0
    s_list = []
    change_lists = []
    E_lists = []
    
    s_copy = s.copy()
    change_list = []
    E_list = []
    E_s = predictor.GetEnergy(s).item()
    E_list.append(E_s)
    for j in tqdm(range(5000)):
        s_new = s_copy.copy()
        atom_random = random.sample(range(len(s_new)), 2)
        s_new[atom_random[0]], s_new[atom_random[1]] = s_new[atom_random[1]], s_new[atom_random[0]]
        E_s_new = predictor.GetEnergy(s_new).item()
        p = math.exp((E_s - E_s_new)/((8.617333262*1e-5) * temperature_initial))
        if p >= random.random():
            s_copy = s_new
            E_s = E_s_new
            E_list.append(E_s_new)
            change_list.append(atom_random)
        else:
            E_list.append(E_s)
            change_list.append(None)
    
    for k in tqdm(range(kmax)):
        temperature = (temperature_initial-temperature_final) * (1- (k+1)/kmax) + temperature_final
        s_new = s_copy.copy()
        atom_random = random.sample(range(len(s_new)), 2)
        s_new[atom_random[0]], s_new[atom_random[1]] = s_new[atom_random[1]], s_new[atom_random[0]]
        E_s_new = predictor.GetEnergy(s_new).item()
        p = math.exp((E_s - E_s_new)/((8.617333262*1e-5) * temperature))
        if p >= random.random():
            s_copy = s_new
            E_s = E_s_new
            E_list.append(E_s_new)
            change_list.append(atom_random)
        else:
            E_list.append(E_s)
            change_list.append(None)

    json.dump((initial_atomic_numbers),open('initial_atoms.json','w'))
    json.dump((E_list),open('E_lists.json','w'))
    json.dump((change_list),open('change_lists.json','w'))
    json.dump((s_copy.tolist()),open('final_atoms.json','w'))
    
    return s_copy.tolist()

##################################
elements = list(map(int, input("Please enter 5 integers: ").split()))
total = sum(elements)
normalized_list = [512 * x / total for x in elements]
counts = [int(x) for x in normalized_list]

atomic_numbers = [44] * counts[0] + [45] * counts[1] +[46] * counts[2] +[77] * counts[3] +[78] * counts[4]
if len(atomic_numbers) < 512:
    atomic_numbers += [78] * (512 - len(atomic_numbers))

shuffle(atomic_numbers)
##################################
atoms = fcc111('Pt', size=(8, 8, 8) , a=3.9672029988415840, vacuum=10)
atoms.set_atomic_numbers(atomic_numbers)

pos = atoms.get_positions()
# find number of layers
zs = pos[:,2]
zs_sorted = np.sort(zs)
natom_in_layer = np.sum(np.abs(zs - zs[-1]) <1)

# set tags to indicate the layer number
z_sorted_index = np.argsort(zs)[::-1]
top_index = z_sorted_index[:natom_in_layer]
tags = np.zeros(len(atoms))
for i in range(int(len(atoms)/natom_in_layer)):
    tags[z_sorted_index[i*natom_in_layer:(i+1)*natom_in_layer]] = i
atoms.set_tags(tags)

# set up neighbor list
nl = NeighborList([1.5]*len(atoms),self_interaction=False,bothways=True)
nl.update(atoms)

# linear model
an_map = [77, 46, 78, 45, 44]
coeff = [0.490,1.178,0.988,0.744,0.329,0.027,-0.022,0.023,-0.017,-0.012,-0.009,0.010,-0.015,0.005,0.008]

zone2_index = [[] for _ in range(len(top_index))]
zone3_index = [[] for _ in range(len(top_index))]

for i, OH_site_index in enumerate(top_index):
    indices,_ = nl.get_neighbors(OH_site_index)
    zone2_index[i] = indices[tags[indices]==0].tolist()
    zone3_index[i] = indices[tags[indices]==1].tolist()


mc = input("Please choose whether or not to perform MC annealing (y/n): ")
if mc == 'y':
    atomic_numbers = GetOptimizedSlab(atoms)
    atoms.set_atomic_numbers(atomic_numbers)

activities = []

for i, OH_site_index in enumerate(top_index):
    zone1 = [0 for _ in range(5)]
    zone2 = [0 for _ in range(5)]
    zone3 = [0 for _ in range(5)]
    zone1[an_map.index(atomic_numbers[OH_site_index])] += 1
    for j in zone2_index[i]:
        zone2[an_map.index(atomic_numbers[j])] += 1
    for j in zone3_index[i]:
        zone3[an_map.index(atomic_numbers[j])] += 1
    
    descriptor_vector = zone1+zone2+zone3
    OH_BE = np.dot(coeff,descriptor_vector)
    activity = (1.38*1e-23)*(298.0)/(6.626*1e-34)*np.exp((-1)*abs(OH_BE-0.895)/((8.617*1e-5)*(298.0)))
    activities.append(activity)

slab_activity = np.log(sum(activities)/len(activities))
print(slab_activity)