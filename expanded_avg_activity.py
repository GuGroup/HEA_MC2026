import json
from ase.build import fcc111
from ase.neighborlist import NeighborList
import numpy as np
from tqdm import tqdm
import copy


atoms = fcc111('Pt', size=(8, 8, 8) , a=3.9672029988415840, vacuum=10)
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
coeff = [0.490,1.178,0.988,0.744,0.329,0.027,-0.022,0.023,-0.017,-0.012,-0.009,0.010,-0.015,0.005,0.008] #시험용

zone2_index = [[] for _ in range(len(top_index))]
zone3_index = [[] for _ in range(len(top_index))]

for i, OH_site_index in enumerate(top_index):
    indices,_ = nl.get_neighbors(OH_site_index)
    zone2_index[i] = indices[tags[indices]==0].tolist()
    zone3_index[i] = indices[tags[indices]==1].tolist()

num = 0
all_activity = [None] * 3721
 
for file_idx in tqdm(range(3721)):
    activity_1000 = []
    try:
        # s_trajectory =json.load(open(f'expanded/final_atoms_{file_idx}.json','r'))
        # s_trajectory =json.load(open(f'initial2/initial_pixels_{file_idx}.json','r'))
        s_trajectory =json.load(open(f'/home/ktg0829/ktg/output_4000K_5k//final_atoms_{file_idx}.json','r'))
        # s_trajectory =json.load(open(f'/home/ktg0829/ktg/output_4000K_5k//intial_atoms_{file_idx}.json','r'))
        atoms = fcc111('Pt', size=(8, 8, 8) , a=3.9672029988415840, vacuum=10)
        s = s_trajectory[0]
        atoms.set_atomic_numbers(s)
        an = atoms.get_atomic_numbers()
        # activities = []

        for s in s_trajectory:
            atoms.set_atomic_numbers(s)
            an = atoms.get_atomic_numbers()
            activities = []

            for i, OH_site_index in enumerate(top_index):
                zone1 = [0 for _ in range(5)]
                zone2 = [0 for _ in range(5)]
                zone3 = [0 for _ in range(5)]
                zone1[an_map.index(an[OH_site_index])] += 1
                for j in zone2_index[i]:
                    zone2[an_map.index(an[j])] += 1
                for j in zone3_index[i]:
                    zone3[an_map.index(an[j])] += 1
                
                descriptor_vector = zone1+zone2+zone3
                OH_BE = np.dot(coeff,descriptor_vector)
                activity = (1.38*1e-23)*(298.0)/(6.626*1e-34)*np.exp((-1)*abs(OH_BE-0.895)/((8.617*1e-5)*(298.0)))
                activities.append(activity)

            pixel_activity = np.log(sum(activities)/len(activities))
            activity_1000.append(pixel_activity)
            
        avg_activity = sum(activity_1000)/len(activity_1000)
        all_activity[file_idx] = avg_activity
    # all_activity.append(activity_1000)
    except:
        num += 1
        continue
    
print(num)
json.dump(all_activity, open('expanded_avg_activity.json', 'w'))
# json.dump(all_activity, open('initial_expanded_avg_activity.json', 'w'))