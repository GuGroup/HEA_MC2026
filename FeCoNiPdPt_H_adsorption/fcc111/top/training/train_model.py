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
MODEL_METADATA = {'description': '*H adsorption energy linear model for FeCoNiPdPt HEA', 'elements': ['Fe', 'Co', 'Ni', 'Pd', 'Pt'], 'zone_sizes': [1, 6, 3], 'binding_atom': None}
PARAMETER_KEYS = {'zone1': ['Fe', 'Co', 'Ni', 'Pd', 'Pt'], 'zone2': ['Fe', 'Co', 'Ni', 'Pd', 'Pt'], 'zone3': ['Fe', 'Co', 'Ni', 'Pd', 'Pt']}
N_FEATURES = 15
INCLUDE_VALID_INDICES = False
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

def get_zones(contcar_path):
    cell, positions_cart, atom_symbols = parse_poscar(contcar_path)
    h_indices = [i for i, s in enumerate(atom_symbols) if s == 'H']
    metal_indices = [i for i, s in enumerate(atom_symbols) if s != 'H']
    h_pos = np.array(positions_cart)[h_indices[0]]
    metal_pos = np.array(positions_cart)[metal_indices]
    metal_sym = [atom_symbols[i] for i in metal_indices]
    z_vals = metal_pos[:, 2]
    z_layers = classify_layers(z_vals, tol=0.5)
    z_top = z_layers[0]
    z_second = z_layers[1]
    top_mask = np.abs(z_vals - z_top) < 0.5
    second_mask = np.abs(z_vals - z_second) < 0.5
    super_pos = []
    super_sym = []
    super_layer = []
    for n1 in range(3):
        for n2 in range(3):
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
    super_pos = np.array(super_pos)
    super_layer = np.array(super_layer)
    h_pos_center = h_pos + 1 * cell[0] + 1 * cell[1]
    top_idx_super = np.where(super_layer == 'top')[0]
    dist_top = np.linalg.norm(super_pos[top_idx_super] - h_pos_center, axis=1)
    zone1_super_local = np.argmin(dist_top)
    zone1_super_idx = top_idx_super[zone1_super_local]
    zone1_sym = super_sym[zone1_super_idx]
    zone1_pos_super = super_pos[zone1_super_idx]
    top_excl_super = top_idx_super[top_idx_super != zone1_super_idx]
    dist_top_excl = np.linalg.norm(super_pos[top_excl_super] - zone1_pos_super, axis=1)
    zone2_local = np.argsort(dist_top_excl)[:6]
    zone2_indices = top_excl_super[zone2_local]
    zone2_syms = [super_sym[i] for i in zone2_indices]
    second_idx_super = np.where(super_layer == 'second')[0]
    dist_second = np.linalg.norm(super_pos[second_idx_super] - zone1_pos_super, axis=1)
    zone3_local = np.argsort(dist_second)[:3]
    zone3_indices = second_idx_super[zone3_local]
    zone3_syms = [super_sym[i] for i in zone3_indices]
    return (zone1_sym, zone2_syms, zone3_syms)

def build_feature_vector(zone1_sym, zone2_syms, zone3_syms):
    features = []
    for el in ELEMENTS:
        features.append(1.0 if zone1_sym == el else 0.0)
    for el in ELEMENTS:
        features.append(float(zone2_syms.count(el)))
    for el in ELEMENTS:
        features.append(float(zone3_syms.count(el)))
    return features

def feature_vector(atoms):
    return build_feature_vector(*get_zones(atoms))

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
