from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import os
import pandas as pd
import matplotlib as mpl

ELEMENTS = ['Ru', 'Rh', 'Pd', 'Ir', 'Pt']

MARKER_RULES = {
    1: ('first index', 0.0),
    2: ('last index', 1.0),
    3: ('middle index', 0.5),
    4: ('1/3 index', 1.0 / 3.0),
    5: ('2/3 index', 2.0 / 3.0),
    6: ('3/4 index', 3.0 / 4.0),
}

def get_selected_position(data, ml_idx):
    valid_positions = np.argwhere(~np.isnan(data))
    if valid_positions.size == 0:
        return None, None, None

    label, fraction = MARKER_RULES.get(ml_idx, ('first index', 0.0))
    selected_idx = int((len(valid_positions) - 1) * fraction)
    row, col = valid_positions[selected_idx]
    return selected_idx, row, col

def get_selected_position_from_alpha(pixel_values, ml_idx):
    valid_positions = np.argwhere(pixel_values[:, :, 3] > 0)
    if valid_positions.size == 0:
        return None, None, None

    label, fraction = MARKER_RULES.get(ml_idx, ('first index', 0.0))
    selected_idx = int((len(valid_positions) - 1) * fraction)
    row, col = valid_positions[selected_idx]
    return selected_idx, row, col

def mark_selected_position(ax, data, ml_idx):
    selected_idx, row, col = get_selected_position(data, ml_idx)
    if row is None:
        return None

    label, _ = MARKER_RULES.get(ml_idx, ('first index', 0.0))
    ax.scatter(
        col, row,
        s=450,
        facecolors='none',
        edgecolors='red',
        linewidths=2.5,
        label=f'{label}: index={selected_idx}, Row_Idx={row}, Col_Idx={col}'
    )
    #ax.legend(loc='upper right')
    return selected_idx, row, col

def mark_selected_position_from_alpha(ax, pixel_values, ml_idx):
    selected_idx, row, col = get_selected_position_from_alpha(pixel_values, ml_idx)
    if row is None:
        return None

    label, _ = MARKER_RULES.get(ml_idx, ('first index', 0.0))
    ax.scatter(
        col, row,
        s=450,
        facecolors='none',
        edgecolors='red',
        linewidths=2.5,
        label=f'{label}: index={selected_idx}, Row_Idx={row}, Col_Idx={col}'
    )
    #ax.legend(loc='upper right')
    return selected_idx, row, col

def extract_pixel_values(cord, img, ml_idx, elem):
    start = cord[0]
    end = cord[1]
    
    x = np.linspace(start[0], end[0], num=22).astype(int)
    y = np.linspace(start[1], end[1], num=21).astype(int)
    
    pixel_values = np.zeros((len(y)-1, len(x) - 1, 4))
    
    mask_rules = {
        1: 5, 2: 4, 3: 3, 4: 2, 5: 1, 6: 1, 
        14: 1, 15: 1, 16: 2, 17: 3, 18: 4, 19: 5, 20: 7
    }
    
    valid_grid_mask = np.zeros((20, 21), dtype=bool)

    for i in range(len(y)-1):
        row_num = i + 1
        side_mask = mask_rules.get(row_num, 0)
        
        for j in range(len(x) - 1):
            if j < side_mask or j >= (len(x) - 1 - side_mask):
                pixel_values[i, j] = [255, 255, 255, 0]
                continue
            
            if j >= side_mask and j < (21 - side_mask):
                valid_grid_mask[i, j] = True
                
            region = img[y[i+1]:y[i], x[j]:x[j+1]]

            # 가장 진한 색상 찾기
            brightness = np.sum(region[:, :, :3], axis=2)
            min_idx = np.unravel_index(np.argmin(brightness), brightness.shape)
            representative_rgb = region[min_idx]
            
            if representative_rgb.shape[0] == 4:
                pixel_values[i, j] = representative_rgb
            else:
                pixel_values[i, j] = np.append(representative_rgb, 255)
    
    pixel_values = np.flipud(pixel_values)
    pixel_values /= 255.0
    
    # np.save('valid_grid_mask.npy', valid_grid_mask)
    # print("saved valid_grid_mask.npy)")
    
    # print(pixel_values.shape)
    
    #fig, ax = plt.subplots(figsize=(6, 6))
    #ax.imshow(pixel_values)
    #ax.set_title(f'ML{ml_idx}.png - {elem}: extracted PNG colors')
    #ax.set_xlabel('Col_Idx')
    #ax.set_ylabel('Row_Idx')
    #mark_selected_position_from_alpha(ax, pixel_values, ml_idx)
    #plt.show()
    
    return pixel_values

def map_viridis_percent(pixel_values, vmin, vmax, ml_idx, elem, ax=None, cmap_name='viridis'):
    res = 256
    colormap = mpl.colormaps[cmap_name]
    cmap_rgb = colormap(np.linspace(0, 1, res))[:, :3]
    
    tree = KDTree(cmap_rgb)
    
    h, w, _ = pixel_values.shape
    rgb_flat = pixel_values[:, :, :3].reshape(-1, 3)
    alpha_flat = pixel_values[:, :, 3].reshape(-1)
    
    converted_values = []
    
    for rgb, alpha in zip(rgb_flat, alpha_flat):
        if alpha == 0: 
            converted_values.append(np.nan)
        else:
            dist, idx = tree.query(rgb)
            val = vmin + (idx / (res - 1)) * (vmax - vmin)
            converted_values.append(val)
            
    converted_np = np.array(converted_values).reshape(h, w)
    
    show_single_plot = ax is None
    if show_single_plot:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    im = ax.imshow(converted_np)
    ax.set_title(f'{elem} / at.%')
    ax.set_xlabel('Col_Idx')
    ax.set_ylabel('Row_Idx')
    mark_selected_position(ax, converted_np, ml_idx)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if show_single_plot:
        plt.show()
        plt.close(fig)
    
    return converted_np
    
def convert_to_512_atoms(ratios):
    if np.any(np.isnan(ratios)):
        return [np.nan] * len(ratios)
    
    counts = np.round(ratios / 100.0 * 512).astype(int)
    
    residual = 512 - np.sum(counts)
    if residual != 0:
        errors = (ratios / 100.0 * 512) - counts
        adjust_idx = np.argmax(np.abs(errors)) if residual > 0 else np.argmin(errors)
        counts[adjust_idx] += residual
        
    return counts


def print_selected_composition(ml_idx, total_sum, normalized_ratios):
    selected_idx, row, col = get_selected_position(total_sum, ml_idx)
    if row is None:
        print(f'ML{ml_idx}: 선택 가능한 유효 픽셀이 없습니다.')
        return

    label, _ = MARKER_RULES.get(ml_idx, ('first index', 0.0))
    ratios = {elem: normalized_ratios[elem][row, col] for elem in ELEMENTS}
    ratio_text = ', '.join(f'{elem}={ratios[elem]:.4f} at.%' for elem in ELEMENTS)
    print(
        f'ML{ml_idx} selected pixel ({label}, index={selected_idx}, '
        f'Row_Idx={row}, Col_Idx={col}): {ratio_text}'
    )

# [x start, y start] [x end, y end]
cords = {
    'Ru': [[83, 295], [348, 41]],
    'Rh': [[502, 295], [767, 41]],
    'Pd': [[921, 295], [1186, 41]],
    'Ir': [[83, 639], [348, 385]],
    'Pt': [[502, 639], [767, 385]]
}

ml_limits = {
    1: {
        'Ru': (11.5, 32.5), 'Rh': (9.2, 37.5), 'Pd': (9.5, 35.0), 
        'Ir': (11.4, 32.7), 'Pt': (11.4, 32.1)
    },
    2: {
        'Ru': (8.9, 28.1), 'Rh': (6.3, 30.9), 'Pd': (11.4, 34.1), 
        'Ir': (11.8, 40), 'Pt': (12.4, 40.6)
    },
    3: {
        'Ru': (12.0, 31.3), 'Rh': (10.0, 37.1), 'Pd': (8.3, 29.6), 
        'Ir': (10.9, 34.6), 'Pt': (13.9, 39.9)
    },
    4: {
        'Ru': (9.7, 30.9), 'Rh': (10.5, 35.7), 'Pd': (11.8, 37.2), 
        'Ir': (9.9, 29.1), 'Pt': (12.7, 33.9)
    },
    5: {
        'Ru': (11.5, 30.9), 'Rh': (12.6, 41.1), 'Pd': (9.6, 31.9), 
        'Ir': (8.6, 30.8), 'Pt': (11.8, 33.9)
    },
    6: {
        'Ru': (10.5, 36.5), 'Rh': (7.2, 35.3), 'Pd': (13.4, 35.9), 
        'Ir': (8.7, 32.3), 'Pt': (8.4, 34.5)
    }
}

for ml_idx in range(1, 7):
    img_name = f'ML{ml_idx}.png'
    if not os.path.exists(img_name):
        print(f"{img_name} 파일이 없어 건너뜁니다.")
        continue
        
    print(f"--- Processing {img_name} ---")
    img = np.asarray(Image.open(img_name))
    
    extracted_ratios = {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes_flat = axes.ravel()
    
    for elem_idx, elem in enumerate(ELEMENTS):
        pixel_vals = extract_pixel_values(cords[elem], img, ml_idx, elem)
        
        vmin, vmax = ml_limits[ml_idx][elem]
        
        extracted_ratios[elem] = map_viridis_percent(
            pixel_vals, vmin, vmax, ml_idx, elem, ax=axes_flat[elem_idx]
        )

    axes_flat[-1].axis('off')
    fig.suptitle(f'{img_name}: viridis-mapped composition (%)')
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    
    total_sum = sum(extracted_ratios.values())
    
    normalized_ratios = {}
    for elem in ELEMENTS:
        normalized_ratios[elem] = (extracted_ratios[elem] / total_sum) * 100

    print_selected_composition(ml_idx, total_sum, normalized_ratios)

    h, w = total_sum.shape
    data_list = []
    
    for i in range(h):
        for j in range(w):
            if np.isnan(total_sum[i, j]):
                continue
                
            pt_ratios = np.array([normalized_ratios[elem][i, j] for elem in ELEMENTS])
            
            pt_counts = convert_to_512_atoms(pt_ratios)
            
            row_data = {
                'Row_Idx': i,
                'Col_Idx': j,
                'Ru_ratio': pt_ratios[0], 'Rh_ratio': pt_ratios[1], 'Pd_ratio': pt_ratios[2],
                'Ir_ratio': pt_ratios[3], 'Pt_ratio': pt_ratios[4],
                'Ru_count': pt_counts[0], 'Rh_count': pt_counts[1], 'Pd_count': pt_counts[2],
                'Ir_count': pt_counts[3], 'Pt_count': pt_counts[4]
            }
            data_list.append(row_data)
            
    df = pd.DataFrame(data_list)
    
    df.to_csv(f'ML{ml_idx}_composition_data.csv', index=False)
    print(f"Saved ML{ml_idx}_composition_data.csv (Num of valid compositions: {len(df)})")
