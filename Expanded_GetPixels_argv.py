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
import numpy as np
from matplotlib import colormaps
from ase.build import fcc111
from ase.visualize import view
from random import shuffle
import gc
from tqdm import tqdm
import sys


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
    json.dump((initial_atomic_numbers),open(f'output_4000K_5k/intial_atoms_{i}.json','w'))
    json.dump((E_lists),open(f'output_4000K_5k/E_lists_{i}.json','w'))
    json.dump((change_lists),open(f'output_4000K_5k/change_lists_{i}.json','w'))
    json.dump((s_list),open(f'output_4000K_5k/final_atoms_{i}.json','w'))

def gaussian(x,y,mux,muy,sigma):
    return np.exp(-((x-mux)**2+(y-muy)**2)/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

def find_overlap_pixels(pixel_val_dict):
    # overlap_pixels_list_51x51 = []
    overlap_pixels_list = []
    # num = 0
    
    x,y = np.meshgrid(np.arange(pixel_val_dict['HER'].shape[1]),np.arange(pixel_val_dict['HER'].shape[0]))
    gauss_vals = np.zeros(pixel_val_dict['HER'].shape[:2])
    gauss_vals[::,49:] += 1
    gauss_vals[49:,:49] += 1
    for pixel_x in range(0,x1.shape[0]-2):
        for pixel_y in range(0,y1.shape[0]-2):            
            if (pixel_values[pixel_y, pixel_x] > 0.92).all():
                gauss_vals[pixel_y, pixel_x] += 1
            if (pixel_values[pixel_y, pixel_x] == 0).all():
                gauss_vals[pixel_y, pixel_x] += 1
    
    padding = 5
    new_size = 51 + 2 * padding # 61
    
    for y in range(51):
        for x in range(51):
            if gauss_vals[y, x] == 0:
                new_y = y + padding
                new_x = x + padding
                
                new_idx = new_y * new_size + new_x
                overlap_pixels_list.append(new_idx)
                
    return overlap_pixels_list

def save_pixel_val_dic(k, start1, end1):
    x1 = np.linspace(start1[0],end1[0],num=51)
    y1 = np.linspace(start1[1],end1[1],num=51)
    xs1,ys1 = np.meshgrid(x1,y1)
    xs1 = xs1.reshape(-1)
    ys1 = ys1.reshape(-1)
    # plt.figure()
    # plt.imshow(img,interpolation='none')
    # plt.scatter(xs1,ys1,s=10,c='k')
     
    # get pixel value.
    x1 = np.round(x1).astype(int)
    y1 = np.round(y1).astype(int)
    pixel_values = np.zeros((y1.shape[0],x1.shape[0],4))

    for pixel_x in range(0,x1.shape[0]-1):
        for pixel_y in range(0,y1.shape[0]-1):
            pixel_values[y1.shape[0]-pixel_y-1,pixel_x] = img[y1[pixel_y+1]:y1[pixel_y],x1[pixel_x]:x1[pixel_x+1]].mean(axis=0).mean(axis=0)
            
    pixel_values /=255
    
    padding = 5
    new_size = 51 + 2 * padding
    
    pixel_values_expanded = np.ones((new_size, new_size, 4))
    pixel_values_expanded[padding : padding + 51, padding : padding + 51] = pixel_values
    # plt.figure()
    # plt.imshow(pixel_values)

    # save
    pixel_val_dict[k] = pixel_values_expanded
    pixel_cross_dict[k] = [25 + padding, 23 + padding]
    
def save_computed_val_dic(k, color, sigma, mus, pixel_val_dict):
    new_size = pixel_val_dict[k].shape[0]
    x,y = np.meshgrid(np.arange(new_size),np.arange(new_size))
    
    padding = 5
    mus_expanded = [(mx + padding, my + padding) for mx, my in mus]
    
    gauss_vals = np.zeros(pixel_val_dict[k].shape[:2])
    for mux,muy in mus_expanded:
        gau_val = gaussian(x,y,mux,muy,sigma)
        gau_val *= sigma*np.sqrt(2*np.pi)
        gauss_vals += gau_val

    computed_val = colormaps[color](gauss_vals)
                
    computed_values[k] = computed_val

    diff = (computed_values[k]- pixel_val_dict[k])
    diff[pixel_cross_dict[k][0],:] = 0
    diff[:,pixel_cross_dict[k][1]] = 0
    rmse = np.sqrt(np.mean(diff**2))

    # plt.figure()
    # plt.imshow(computed_val)

    diff = (computed_values[k]- pixel_val_dict[k])
    diff[pixel_cross_dict[k][0],:] = 0
    diff[:,pixel_cross_dict[k][1]] = 0
    rmse = np.sqrt(np.mean(diff**2))
    rmse_values[k] = rmse

if __name__ == '__main__':
    run_idx = int(sys.argv[1])   
    print('running %i'%run_idx)
    # initialize
    
    best_y, best_x = 1, 4
    new_size = 61
    padding = 5
    
    im_frame = Image.open('Untitled.png')
    img = np.asarray(im_frame)
    # plt.figure()
    # plt.imshow(img,interpolation='none')
    pixel_val_dict = {}
    pixel_cross_dict = {}
    rmse_values = {}

    #HER
    im_frame2 = Image.open('activitymap.png')
    img2 = np.asarray(im_frame2)

    start1 = [86,426]
    end1 = [436,78]
    x1 = np.linspace(start1[0],end1[0],num=51)
    y1 = np.linspace(start1[1],end1[1],num=51)
    xs1,ys1 = np.meshgrid(x1,y1)
    xs1 = xs1.reshape(-1)
    ys1 = ys1.reshape(-1)

    x1 = np.round(x1).astype(int)
    y1 = np.round(y1).astype(int)
    pixel_values = np.zeros((y1.shape[0],x1.shape[0],4))

    for pixel_x in range(0,x1.shape[0]-1):
        for pixel_y in range(0,y1.shape[0]-1):
            pixel_values[y1.shape[0]-pixel_y-1,pixel_x] = img2[y1[pixel_y+1]:y1[pixel_y],x1[pixel_x]:x1[pixel_x+1]].mean(axis=0).mean(axis=0)
            
    pixel_values /=255
    # plt.figure()
    # plt.imshow(pixel_values)

    pixel_val_dict['HER'] = pixel_values

    x,y = np.meshgrid(np.arange(pixel_val_dict['HER'].shape[1]),np.arange(pixel_val_dict['HER'].shape[0]))
    gauss_vals = np.zeros(pixel_val_dict['HER'].shape[:2])

    # Check pixel corner location

    #Pd
    start1 = [0,287]
    end1 = [261,30]
    save_pixel_val_dic('Pd', start1, end1)

    #Ir
    start1 = [288,287]
    end1 = [549,30]
    save_pixel_val_dic('Ir', start1, end1)

    #Rh
    start1 = [557,287]
    end1 = [818,30]
    save_pixel_val_dic('Rh', start1, end1)

    #Ru
    start1 = [842,287]
    end1 = [1103,30]
    save_pixel_val_dic('Ru', start1, end1)

    #Pt
    start1 = [1114,287]
    end1 = [1375,30]
    save_pixel_val_dic('Pt', start1, end1)

    # Gaussian values
    computed_values = {}

    ## Pd
    sigma = 2.602
    mus = [
           [25.5,8.5],[35.5,11.3],[42.5,19.5],[43.5,30.5],[37.5,39.5],[27.5,44.5],[17.4,41.4],[10.5,33.5],[10.5,22.4],[15.5,12.4],[31.5,19.5],[34.8,28.8],[26.8,34.5],[19.6,29.8],[21.5,20.4],[27.4, 26.3]
           ]
    save_computed_val_dic('Pd', 'Reds', sigma, mus, pixel_val_dict)

    ##Ir
    sigma = 2.391
    mus = [
           [25.5,5.5],[35.5,8.5],[42.5,16.5],[43.5,27.5],[37.5,36.5],[27.7,41.4],[18.4,38.5],[11.7,30.5],[9.7,19.4],[15.5,9.5],[30.6,15.5],[34.6,24.8],[27.6,31.3],[20.6,25.6],[21.6,17.4],[27.5,22.5]
           ]
    save_computed_val_dic('Ir', 'Greens', sigma, mus, pixel_val_dict)

    #Rh
    sigma = 2.668
    mus = [
           [23.5,6.5],[33.5,9.5],[40.5,17.5],[40.5,28.5],[35.5,38.5],[25.5,41.5],[15.5,39.5],[8.5,31.5],[8.5,20.5],[13.5,10.5],[28.5,17.4],[31.6,26.5],[24.9,32.2],[17.3,27.5],[19.5,17.6],[24.5,24.4]
           ]
    save_computed_val_dic('Rh', 'Purples', sigma, mus, pixel_val_dict)

    #Ru
    sigma = 2.599
    mus = [
           [24.5,5.5],[34.5,7.5],[41.5,16.5],[42.5,27.5],[36.5,36.5],[26.5,40.5],[16.5,38.5],[9.5,29.5],[9.5,19.5],[14.5,10.5],[29.6,16.4],[33.5,24.5],[26.5,30.5],[18.5,25.5],[20.5,16.5],[25.5,22.5]
           ]
    save_computed_val_dic('Ru', 'Blues', sigma, mus, pixel_val_dict)

    #Pt
    sigma = 2.490
    mus = [
           [24.5,9.5],[35.4,11.5],[41.5,20.5],[42.5,31.5],[36.3,40.6],[27.5,44.5],[17.5,42.5],[10.5,33.5],[9.5,22.5],[15.5,13.5],[30.4,20.5],[33.5,29.5],[26.5,35.5],[18.5,30.5],[20.5,20.5],[25.5,27.5]
           ]
    save_computed_val_dic('Pt', 'Oranges', sigma, mus, pixel_val_dict)

    #print rmse
    print(rmse_values)

    total_gauss_vals = np.zeros((new_size, new_size))
    
    for val in computed_values.values():
        total_gauss_vals += val[:, :, 0]
        
    for i, val in enumerate(total_gauss_vals):
        for j, v in enumerate(val):
            if v == 0:
                total_gauss_vals[i][j] = 1
            
    normalized_gauss_vals = {}
    for key, val in computed_values.items():
        normalized_gauss_vals[key] = val[:, :, 0] / total_gauss_vals

    #for key, val in normalized_gauss_vals.items():
    #    plt.figure()
    #    plt.title(f"Normalized values for {key}")
    #    plt.imshow(val, cmap='viridis')
    #    plt.colorbar()

    atoms = [[[] for _ in range (new_size)] for _ in range(new_size)]

    for key, value in normalized_gauss_vals.items():
        if key == 'Pd':
            a = 46
        elif key == 'Ir':
            a = 77
        elif key == 'Rh':
            a = 45
        elif key == 'Ru':
            a = 44
        else:
            a = 78
            
        for i in range(new_size):
            for j in range(new_size):
                count = int(512 * value[i, j])
                if count > 0:
                    for _ in range(count):
                        atoms[i][j].append(a)
                
    pixels = []
      
    for i in range(new_size):
        for j in range(new_size):
            if len(atoms[i][j]) > 0:
                if len(atoms[i][j]) < 512:
                    for _ in range(512-len(atoms[i][j])):
                        atoms[i][j].append(78)
                pixel = fcc111('Pt', size=(8, 8, 8) , a=3.9672029988415840, vacuum=10)
                atoms_numbers = atoms[i][j]
                shuffle(atoms_numbers)
                pixel.set_atomic_numbers(atoms_numbers)
                pixels.append(pixel)
                
    os.environ['VASP_PP_PATH']="/home/shared/programs/vasp/vasp_pp"
    '''
    for i, slab in enumerate(pixels):
        s = slab.get_atomic_numbers()
        predictor = Predictor4MC(slab,'model_best.pth.tar','atom_init.json')
        E_s = predictor.GetEnergy(s)

        #print('initial_state:', s)
        kmax = 500000
        temperature_initial = 1000.0
        temperature_final = 298.0

        s_trajectory = [s]
        T_trajectory = [temperature_initial]
        E_trajectory = [E_s]
        for k in tqdm(range(kmax)):
            temperature = (temperature_initial-temperature_final) * (1- (k+1)/kmax) + temperature_final
            slab_new = slab.copy()
            s_new = s.copy()
            atom_random = random.sample(range(len(s_new)), 2)
            s_new[atom_random[0]], s_new[atom_random[1]] = s_new[atom_random[1]], s_new[atom_random[0]]
            E_s_new = predictor.GetEnergy(s_new)
            p = math.exp((E_s - E_s_new)/((8.617333262*1e-5) * temperature))
            if p >= random.random():
                slab = slab_new
                s = s_new
                E_s = E_s_new
            else:
                pass
            E_trajectory.append(E_s)
            s_trajectory.append(s)
            T_trajectory.append(temperature)
        #print('Final state:', s)

        #import matplotlib.pyplot as plt

        #plt.scatter(list(range(len(E_trajectory))),E_trajectory,s=1)
        #plt.scatter(list(range(len(E_trajectory))),T_trajectory,s=1)
        #plt.show()
        #view(slab)

        s_trajectory = np.array(s_trajectory).tolist()
        T_trajectory = np.array(T_trajectory).tolist()
        E_trajectory = np.array(E_trajectory).tolist()

        write(f'final_{i}.CONTCAR',slab)
        json.dump((s_trajectory,E_trajectory,T_trajectory),open(f'trajectory_{i}.json','w'))
    '''
    inputs = [(i,slab) for i, slab in enumerate(pixels)]
    inputs = [inputs[run_idx]]
    ParalProcess(GetOptimizedSlab,inputs, processes=1)

    '''
    rmses = []
    sigmas = np.linspace(2.0,3.0)
    for sigma in sigmas:
        mus = [
                [24.5,9.5],
                [35.4,11.5],
                [41.5,20.5],
                [42.5,31.5],
                [36.3,40.6],
                [27.5,44.5],
                [17.5,42.5],
                [10.5,33.5],
                [9.5,22.5],
                [15.5,13.5],
                [30.4,20.5],
                [33.5,29.5],
                [26.5,35.5],
                [18.5,30.5],
                [20.5,20.5],
                [25.5,27.5]
                ]
        gauss_vals = np.zeros(pixel_val_dict['Pt'].shape[:2])
        for mux,muy in mus:
            gau_val = gaussian(x,y,mux,muy,sigma)
            gau_val *= sigma*np.sqrt(2*np.pi)
            gauss_vals += gau_val
        
        computed_val = colormaps['Oranges'](gauss_vals)
        k = 'Pt'
        diff = (computed_val- pixel_val_dict[k])
        diff[pixel_cross_dict[k][0],:] = 0
        diff[:,pixel_cross_dict[k][1]] = 0
        rmse = np.sqrt(np.mean(diff**2))
        rmses.append(rmse)
    '''
