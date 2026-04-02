from PIL import Image
import numpy as np
import json
from matplotlib import cm
import os

save_dir = '../Data/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

im_frame2 = Image.open('activitymap.png')
img2 = np.asarray(im_frame2)
value = [[[] for _ in range (51)] for _ in range(51)]
value_list = []

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

for i in range(len(value)):
    value[i][49:] = [0] * len(value[i][49:])
    
for i in range(49, len(value)):
    value[i] = [0] * len(value[i])

for pixel_x in range(0,x1.shape[0]-2):
    for pixel_y in range(0,y1.shape[0]-2):
        if (pixel_values[pixel_y, pixel_x] > 0.92).all():
            value[pixel_y][pixel_x]=0
        if (pixel_values[pixel_y, pixel_x] == 0).all():
            value[pixel_y][pixel_x]=0

start1 = [595,426]
end1 = [945,78]
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

ngrid = 1000
x = np.linspace(-25, -11, ngrid)
y = cm.bwr(1 - (x + 25) / 14)

for i in range(51):
    for j in range(51):
        exp = pixel_values[i][j][:-1]
        dist = np.sqrt(np.sum((y[:, :3] - exp) ** 2, axis=1))
        if value[i][j] == 0:
            pass
        else:
            index = np.argmin(dist)
            value[i][j] = x[index]
            value_list.append(value[i][j])

value_array = np.array(value, dtype=float)

valid_mask = (value_array != 0)
exp_data_valid = value_array[valid_mask]
exp_log = np.log(exp_data_valid - np.min(exp_data_valid) + 1)

exp_min, exp_max = np.min(exp_log), np.max(exp_log)
exp_scaled_values = (exp_log - exp_min) / (exp_max - exp_min)

exp_map_viz = np.full((51, 51), np.nan)
exp_map_viz[valid_mask] = exp_scaled_values

json.dump(exp_scaled_values.tolist(), open(os.path.join(save_dir, 'exp_scaled_activity_1D.json'), 'w'))
json.dump(exp_map_viz.tolist(), open(os.path.join(save_dir, 'exp_scaled_activity_2D.json'), 'w'))