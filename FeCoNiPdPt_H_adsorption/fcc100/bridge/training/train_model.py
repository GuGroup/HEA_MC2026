#!/usr/bin/env python3
"""Train only from adjacent ASE training_data.json, preserving original features.
Run: python train_model.py [--data DATA.json] [--output MODEL.json]
Dependencies: see the top-level requirements.txt.
All records are fitted with LinearRegression(fit_intercept=False).
Reported MAE/RMSE are in-sample; independent evaluation is in ../testing.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io.jsonio import decode
from sklearn.linear_model import LinearRegression

# Adapter keeps legacy zone calculations unchanged while removing file dependencies.
def parse_poscar(atoms):
    return np.asarray(atoms.cell), atoms.positions, atoms.get_chemical_symbols()

ELEMENTS = ['Fe', 'Co', 'Ni', 'Pd', 'Pt']
ZONE1_COMBOS = list(itertools.combinations_with_replacement(ELEMENTS, 2))
ZONE1_COMBO_INDEX = {combo: i for i, combo in enumerate(ZONE1_COMBOS)}
ZONE3_COMBOS = list(itertools.combinations_with_replacement(ELEMENTS, 2))
ZONE3_COMBO_INDEX = {combo: i for i, combo in enumerate(ZONE3_COMBOS)}
MODEL_METADATA = {'description': '*H (fcc(100) bridge) adsorption energy linear model for FeCoNiPdPt HEA', 'elements': ['Fe', 'Co', 'Ni', 'Pd', 'Pt'], 'binding_site': 'fcc100_bridge', 'zone_sizes': [2, 6, 2, 4], 'n_parameters': 40}
PARAMETER_KEYS = {'zone1': ['FeFe', 'FeCo', 'FeNi', 'FePd', 'FePt', 'CoCo', 'CoNi', 'CoPd', 'CoPt', 'NiNi', 'NiPd', 'NiPt', 'PdPd', 'PdPt', 'PtPt'], 'zone2': ['Fe', 'Co', 'Ni', 'Pd', 'Pt'], 'zone3': ['FeFe', 'FeCo', 'FeNi', 'FePd', 'FePt', 'CoCo', 'CoNi', 'CoPd', 'CoPt', 'NiNi', 'NiPd', 'NiPt', 'PdPd', 'PdPt', 'PtPt'], 'zone4': ['Fe', 'Co', 'Ni', 'Pd', 'Pt']}
N_FEATURES = 40
INCLUDE_VALID_INDICES = True
def classify_layers(z_vals, tol=0.5):
    z_sorted = np.sort(z_vals)
    layers = []
    current = [z_sorted[0]]
    for z in z_sorted[1:]:
        if z - current[-1] < tol:
            current.append(z)
        else:
            layers.append(np.mean(current))
            current = [z]
    layers.append(np.mean(current))
    return np.sort(layers)[::-1]

def same_position(pos1, pos2, tol=1e-06):
    return np.linalg.norm(pos1 - pos2) < tol

def near_seen(pos, seen_positions, tol=0.5):
    return any((np.linalg.norm(pos - seen) < tol for seen in seen_positions))

def select_images(sorted_indices, super_pos, n, exclude_positions=None, duplicate_tol=0.5):
    if exclude_positions is None:
        exclude_positions = []
    selected = []
    selected_positions = []
    for idx in sorted_indices:
        pos = super_pos[idx]
        if any((same_position(pos, ex) for ex in exclude_positions)):
            continue
        if near_seen(pos, selected_positions, tol=duplicate_tol):
            continue
        selected.append(idx)
        selected_positions.append(pos)
        if len(selected) == n:
            break
    return selected

def build_supercell(cell, metal_pos, metal_sym, top_mask, second_mask, repeat_range):
    super_pos = []
    super_sym = []
    super_layer = []
    for n1 in repeat_range:
        for n2 in repeat_range:
            shift = n1 * cell[0] + n2 * cell[1]
            for i, (pos, sym) in enumerate(zip(metal_pos, metal_sym)):
                super_pos.append(pos + shift)
                super_sym.append(sym)
                if top_mask[i]:
                    super_layer.append('top')
                elif second_mask[i]:
                    super_layer.append('second')
                else:
                    super_layer.append('other')
    return (np.array(super_pos, dtype=float), np.array(super_sym, dtype=object), np.array(super_layer, dtype=object))

def get_zones_fcc100_bridge(contcar_path, layer_tol=0.5, repeat_range=range(-3, 4)):
    cell, positions_cart, atom_symbols = parse_poscar(contcar_path)
    h_indices = [i for i, s in enumerate(atom_symbols) if s == 'H']
    if len(h_indices) != 1:
        raise ValueError(f'expected exactly one H atom, found {len(h_indices)}')
    metal_indices = [i for i, s in enumerate(atom_symbols) if s != 'H']
    metal_pos = positions_cart[metal_indices]
    metal_sym = [atom_symbols[i] for i in metal_indices]
    z_vals = metal_pos[:, 2]
    z_layers = classify_layers(z_vals, tol=layer_tol)
    if len(z_layers) < 2:
        raise ValueError(f'expected at least two metal layers, found {len(z_layers)}')
    z_top = z_layers[0]
    z_second = z_layers[1]
    top_mask = np.abs(z_vals - z_top) < layer_tol
    second_mask = np.abs(z_vals - z_second) < layer_tol
    super_pos, super_sym, super_layer = build_supercell(cell, metal_pos, metal_sym, top_mask, second_mask, repeat_range)
    h_pos = positions_cart[h_indices[0]]
    top_idx = np.where(super_layer == 'top')[0]
    second_idx = np.where(super_layer == 'second')[0]
    top_by_h = top_idx[np.argsort(np.linalg.norm(super_pos[top_idx] - h_pos, axis=1))]
    zone1_idx = select_images(top_by_h, super_pos, n=2, duplicate_tol=layer_tol)
    if len(zone1_idx) < 2:
        raise ValueError(f'zone1 has only {len(zone1_idx)} atoms')
    zone1_pos = [super_pos[i] for i in zone1_idx]
    zone1_syms = [str(super_sym[i]) for i in zone1_idx]
    zone2_idx = []
    for z1i in zone1_idx:
        top_by_z1 = top_idx[np.argsort(np.linalg.norm(super_pos[top_idx] - super_pos[z1i], axis=1))]
        selected = select_images(top_by_z1, super_pos, n=3, exclude_positions=[super_pos[z1i]], duplicate_tol=layer_tol)
        zone2_idx.extend(selected)
    if len(zone2_idx) < 6:
        raise ValueError(f'zone2 has only {len(zone2_idx)} atoms')
    zone2_syms = [str(super_sym[i]) for i in zone2_idx[:6]]
    dist_to_both = np.linalg.norm(super_pos[second_idx] - zone1_pos[0], axis=1) + np.linalg.norm(super_pos[second_idx] - zone1_pos[1], axis=1)
    second_by_bridge = second_idx[np.argsort(dist_to_both)]
    zone3_idx = select_images(second_by_bridge, super_pos, n=2, duplicate_tol=layer_tol)
    if len(zone3_idx) < 2:
        raise ValueError(f'zone3 has only {len(zone3_idx)} atoms')
    zone3_pos = [super_pos[i] for i in zone3_idx]
    zone3_syms = [str(super_sym[i]) for i in zone3_idx]
    zone4_idx = []
    for z1i in zone1_idx:
        second_by_z1 = second_idx[np.argsort(np.linalg.norm(super_pos[second_idx] - super_pos[z1i], axis=1))]
        selected = select_images(second_by_z1, super_pos, n=2, exclude_positions=zone3_pos, duplicate_tol=layer_tol)
        zone4_idx.extend(selected)
    if len(zone4_idx) < 4:
        raise ValueError(f'zone4 has only {len(zone4_idx)} atoms')
    zone4_syms = [str(super_sym[i]) for i in zone4_idx[:4]]
    return (zone1_syms, zone2_syms, zone3_syms, zone4_syms)

def build_feature_vector(zone1_syms, zone2_syms, zone3_syms, zone4_syms):
    zone1_key = tuple(sorted(zone1_syms, key=lambda s: ELEMENTS.index(s)))
    vec1 = [0.0] * len(ZONE1_COMBOS)
    vec1[ZONE1_COMBO_INDEX[zone1_key]] = 1.0
    vec2 = [float(zone2_syms.count(el)) for el in ELEMENTS]
    zone3_key = tuple(sorted(zone3_syms, key=lambda s: ELEMENTS.index(s)))
    vec3 = [0.0] * len(ZONE3_COMBOS)
    vec3[ZONE3_COMBO_INDEX[zone3_key]] = 1.0
    vec4 = [float(zone4_syms.count(el)) for el in ELEMENTS]
    return vec1 + vec2 + vec3 + vec4

def feature_vector(atoms):
    return build_feature_vector(*get_zones_fcc100_bridge(atoms))

def load_dataset(path):
    data = decode(Path(path).read_text(encoding='utf-8'))
    records = data['records']
    if not records or len(records) != data['n_records']:
        raise ValueError('Empty or inconsistent dataset')
    indices, features, targets = [], [], []
    for record in records:
        atoms = record['atoms']
        if not isinstance(atoms, Atoms):
            raise TypeError('Expected ASE Atoms')
        symbols = atoms.get_chemical_symbols()
        if symbols.count('H') != 1 or not set(symbols) <= set(ELEMENTS + ['H']):
            raise ValueError('Expected one H and only alloy elements')
        if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell).all():
            raise ValueError('Non-finite structure')
        indices.append(record['index'])
        features.append(feature_vector(atoms))
        targets.append(float(record['adsorption_energy_eV']))
    X, y = np.asarray(features, dtype=float), np.asarray(targets, dtype=float)
    if len(set(indices)) != len(indices) or not np.isfinite(y).all():
        raise ValueError('Duplicate indices or invalid targets')
    if X.shape != (len(records), N_FEATURES) or not np.isfinite(X).all():
        raise ValueError('Invalid feature matrix')
    return data, X, y


def fit_dataset(data, X, y):
    model = LinearRegression(fit_intercept=False).fit(X, y)
    residual = model.predict(X) - y
    result = dict(MODEL_METADATA)
    result.update(n_total=len(y), mae_eV=round(float(np.mean(np.abs(residual))), 6),
                  rmse_eV=round(float(np.sqrt(np.mean(residual ** 2))), 6),
                  coef=model.coef_.tolist())
    offset = 0
    parameters = {}
    for zone, keys in PARAMETER_KEYS.items():
        parameters[zone] = {key: float(model.coef_[offset + i]) for i, key in enumerate(keys)}
        offset += len(keys)
    assert offset == N_FEATURES
    result['parameters'] = parameters
    if INCLUDE_VALID_INDICES:
        result['valid_indices'] = [int(r['index']) for r in data['records']]
    return result


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=here / 'training_data.json')
    parser.add_argument('--output', type=Path, default=here / 'model.json')
    args = parser.parse_args()
    result = fit_dataset(*load_dataset(args.data))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['n_total', 'mae_eV', 'rmse_eV']}), flush=True)


if __name__ == '__main__':
    main()
