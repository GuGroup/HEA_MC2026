import os
import random
import math
from ase.calculators.vasp import Vasp
from MCPredict import Predictor4MC
from tqdm import tqdm
import json
from ase.io import write
from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt
from matplotlib import colormaps
from ase.build import fcc111
from ase.visualize import view
from random import shuffle
import gc
from tqdm import tqdm
import sys
import itertools


def ParalIter(processes=32, maxtasksperchild=1):
    gc.collect()
    if 'get_ipython' in locals().keys(): # it doesnt work in ipython
        multiprocessing = None
    elif processes == 1:
        mapper = map
    else: 
        try: 
            from multiprocessing import Pool
            p = Pool(processes=processes, maxtasksperchild=maxtasksperchild)
            mapper = p.imap_unordered
        except:
            mapper = map
    
    return mapper

def ParalProcess(func,inputs, processes=32, maxtasksperchild=1):
    mapper = ParalIter(processes, maxtasksperchild)
    return list(tqdm(mapper(func,inputs),total = len(inputs)))

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def generate_composition_grid(step=0.05, n_components=5):
    """
    합이 1이 되는 n개 성분의 그리드를 생성합니다.
    """
    n_steps = int(round(1.0 / step))
    grid = []
    # 중복 조합을 사용하여 합이 n_steps가 되는 정수 조합을 찾음
    for combo in itertools.combinations_with_replacement(range(n_components), n_steps):
        counts = np.zeros(n_components)
        for i in combo:
            counts[i] += 1
        grid.append(counts * step)
    return np.array(grid)

def GetOptimizedSlab(inputs):
    i, slab = inputs
    initial_atomic_numbers = slab.get_atomic_numbers().tolist()
    #write(f'output/first_{i}.CONTCAR',slab)
    s = slab.get_atomic_numbers()
    predictor = Predictor4MC(slab,'model_best.pth.tar','atom_init.json')

    #print('initial_state:', s)
    kmax = 50000
    temperature_initial = 4000.0
    temperature_final = 298.0
    s_list = []
    change_lists = []
    E_lists = []
    
    for n in range(20):
        s_copy = s.copy()
        change_list = []
        E_list = []
        E_s = predictor.GetEnergy(s).item()
        E_list.append(E_s)
        for j in range(5000):
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
        
        for k in range(kmax):
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
        E_lists.append(E_list)
        change_lists.append(change_list)
        s_list.append(s_copy)

    #write(f'output/final_{i}.CONTCAR',slab)
    s_list = np.array(s_list).tolist()
    json.dump((initial_atomic_numbers),open(f'composition_grid/intial_atoms_{i}.json','w'))
    json.dump((E_lists),open(f'composition_grid/E_lists_{i}.json','w'))
    json.dump((change_lists),open(f'composition_grid/change_lists_{i}.json','w'))
    json.dump((s_list),open(f'composition_grid/final_atoms_{i}.json','w'))

if __name__ == '__main__':
    run_idx = int(sys.argv[1])   
    print('running %i'%run_idx)
    # initialize
    
    grid_x = generate_composition_grid(step=0.2, n_components=5)
    
    element_map = [44, 45, 46, 77, 78]
    total_atoms_count = 512

    pixels = []

    for comp in grid_x:
        counts = np.round(comp * total_atoms_count).astype(int)
        
        diff = total_atoms_count - np.sum(counts)
        counts[-1] += diff
        
        atom_numbers = []
        for idx, count in enumerate(counts):
            atom_numbers.extend([element_map[idx]] * count)
        
        shuffle(atom_numbers)
        
        pixel = fcc111('Pt', size=(8, 8, 8), a=3.9672029988415840, vacuum=10)
        pixel.set_atomic_numbers(atom_numbers)
        pixels.append(pixel)
                
    if run_idx < len(pixels):
        inputs = [(run_idx, pixels[run_idx])]
        ParalProcess(GetOptimizedSlab, inputs, processes=1)
    else:
        print(f"Index {run_idx} is out of range (Total grids: {len(pixels)})")