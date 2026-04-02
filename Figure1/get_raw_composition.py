import os
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import colormaps

def gaussian(x,y,mux,muy,sigma):
    return np.exp(-((x-mux)**2+(y-muy)**2)/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

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
        
    # initialize
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

    pixel_val_dict['HER'] = pixel_values

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

    new_size = 61
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
    
    for i in range(new_size):
        for j in range(new_size):
            if len(atoms[i][j]) > 0:
                if len(atoms[i][j]) < 512:
                    for _ in range(512-len(atoms[i][j])):
                        atoms[i][j].append(78)

    atom_to_element = {44: 'Ru', 45: 'Rh', 46: 'Pd', 77: 'Ir', 78: 'Pt'}

    composition_data = []

    for i in range(new_size):
        for j in range(new_size):
            pixel_idx = i * new_size + j
            pixel_atoms = atoms[i][j]
            
            counts = {el: 0 for el in atom_to_element.values()}
            
            for atomic_num in pixel_atoms:
                if atomic_num in atom_to_element:
                    counts[atom_to_element[atomic_num]] += 1
            
            row = {
                'Pixel_Index': pixel_idx,
                'Ru': counts['Ru'] / 512.0,
                'Rh': counts['Rh'] / 512.0,
                'Pd': counts['Pd'] / 512.0,
                'Ir': counts['Ir'] / 512.0,
                'Pt': counts['Pt'] / 512.0
            }
            composition_data.append(row)

    df_composition = pd.DataFrame(composition_data)

    df_composition = df_composition[['Pixel_Index', 'Ru', 'Rh', 'Pd', 'Ir', 'Pt']]
    
    dir_path = '../Data/'
    file_name = 'composition_raw.csv'
    full_path = os.path.join(dir_path, file_name)

    df_composition.to_csv(full_path, index=False)