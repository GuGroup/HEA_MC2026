import numpy as np
import json
import os
import pandas as pd

dir_path = '../Data/'

exp_scaled_activity_1D = json.load(open(os.path.join(dir_path, 'exp_scaled_activity_1D.json'), 'r'))
exp_scaled_activity_2D = json.load(open(os.path.join(dir_path, 'exp_scaled_activity_2D.json'), 'r'))

MC_raw = json.load(open(os.path.join(dir_path, 'before_fitting_MC_activity.json'), 'r'))
RS_raw = json.load(open(os.path.join(dir_path, 'before_fitting_RS_activity.json'), 'r'))
composition_raw = pd.read_csv(os.path.join(dir_path, 'composition_raw.csv'))

MC_raw_array = np.array([x if x is not None else np.nan for x in MC_raw]).reshape(61, 61)
RS_raw_array = np.array([x if x is not None else np.nan for x in RS_raw]).reshape(61, 61)

exp_array = np.array(exp_scaled_activity_2D, dtype=float)
valid_mask = ~np.isnan(exp_array)

differences = []
coordinates = []

for ystart in range(11):
    for xstart in range(11):
        patch = MC_raw_array[ystart:ystart+51, xstart:xstart+51]
        patch_valid = patch[valid_mask]
        
        p_min, p_max = np.min(patch_valid), np.max(patch_valid)
        patch_scaled = (patch_valid - p_min) / (p_max - p_min)
        
        diff = np.mean(np.abs(patch_scaled - exp_scaled_activity_1D))
        
        differences.append(diff)
        coordinates.append((ystart, xstart))

min_diff = min(differences)

best_idx = differences.index(min_diff)

best_y, best_x = coordinates[best_idx]

print(f'Min MAE: {min_diff:.6f}')

print(f'ystart, xstart: ({best_y}, {best_x})')

MC_patch = MC_raw_array[best_y:best_y+51, best_x:best_x+51]
MC_patch_valid = MC_patch[valid_mask]

bp_min, bp_max = np.min(MC_patch_valid), np.max(MC_patch_valid)
if bp_max - bp_min == 0:
     MC_patch_scaled = MC_patch_valid
else:
    MC_patch_scaled = (MC_patch_valid - bp_min) / (bp_max - bp_min)

MC_patch_viz = np.full((51, 51), np.nan)
MC_patch_viz[valid_mask] = MC_patch_scaled


RS_patch = RS_raw_array[best_y:best_y+51, best_x:best_x+51]
RS_patch_valid = RS_patch[valid_mask]

RS_min, RS_max = np.min(RS_patch_valid), np.max(RS_patch_valid)
if RS_max - RS_min == 0:
     RS_patch_scaled = RS_patch_valid
else:
    RS_patch_scaled = (RS_patch_valid - RS_min) / (RS_max - RS_min)
    
RS_patch_viz = np.full((51, 51), np.nan)
RS_patch_viz[valid_mask] = RS_patch_scaled

# json.dump(MC_patch_scaled.tolist(), open(os.path.join(dir_path, 'MC_activity_patch_1D.json'), 'w'))
# json.dump(RS_patch_scaled.tolist(), open(os.path.join(dir_path, 'RS_activity_patch_1D.json'), 'w'))

# json.dump(MC_patch_viz.tolist(), open(os.path.join(dir_path, 'MC_activity_patch_2D.json'), 'w'))
# json.dump(RS_patch_viz.tolist(), open(os.path.join(dir_path, 'RS_activity_patch_2D.json'), 'w'))

global_idx_grid = np.arange(3721).reshape(61, 61)

local_idx_grid = np.arange(51 * 51).reshape(51, 51)

elements = ['Ru', 'Rh', 'Pd', 'Ir', 'Pt']
comp_grids = {el: composition_raw[el].values.reshape(61, 61) for el in elements}

patch_composition_data = {
    'Local_Index': [],
    'Global_Index': [],
    'Ru': [], 'Rh': [], 'Pd': [], 'Ir': [], 'Pt': []
}

for el in elements:
    patch = comp_grids[el][best_y:best_y+51, best_x:best_x+51]
    patch_composition_data[el] = patch[valid_mask]

global_patch = global_idx_grid[best_y:best_y+51, best_x:best_x+51]
patch_composition_data['Global_Index'] = global_patch[valid_mask]

patch_composition_data['Local_Index'] = local_idx_grid[valid_mask]

df_patch_composition = pd.DataFrame(patch_composition_data)

column_order = ['Local_Index', 'Global_Index', 'Ru', 'Rh', 'Pd', 'Ir', 'Pt']
df_patch_composition = df_patch_composition[column_order]

save_path = os.path.join(dir_path, 'composition_patch_1D.csv')
df_patch_composition.to_csv(save_path, index=False)