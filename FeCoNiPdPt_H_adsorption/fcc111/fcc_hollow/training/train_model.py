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
ZONE1_COMBOS = list(itertools.combinations_with_replacement(ELEMENTS, 3))
ZONE1_COMBO_INDEX = {combo: i for i, combo in enumerate(ZONE1_COMBOS)}
MODEL_METADATA = {'description': '*H (fcc hollow) adsorption energy linear model for FeCoNiPdPt HEA', 'elements': ['Fe', 'Co', 'Ni', 'Pd', 'Pt'], 'binding_site': 'fcc_hollow', 'zone_sizes': [3, 6, 3, 3, 3], 'n_parameters': 55}
PARAMETER_KEYS = {'zone1': ['FeFeFe', 'FeFeCo', 'FeFeNi', 'FeFePd', 'FeFePt', 'FeCoCo', 'FeCoNi', 'FeCoPd', 'FeCoPt', 'FeNiNi', 'FeNiPd', 'FeNiPt', 'FePdPd', 'FePdPt', 'FePtPt', 'CoCoCo', 'CoCoNi', 'CoCoPd', 'CoCoPt', 'CoNiNi', 'CoNiPd', 'CoNiPt', 'CoPdPd', 'CoPdPt', 'CoPtPt', 'NiNiNi', 'NiNiPd', 'NiNiPt', 'NiPdPd', 'NiPdPt', 'NiPtPt', 'PdPdPd', 'PdPdPt', 'PdPtPt', 'PtPtPt'], 'zone2': ['Fe', 'Co', 'Ni', 'Pd', 'Pt'], 'zone3': ['Fe', 'Co', 'Ni', 'Pd', 'Pt'], 'zone4': ['Fe', 'Co', 'Ni', 'Pd', 'Pt'], 'zone5': ['Fe', 'Co', 'Ni', 'Pd', 'Pt']}
N_FEATURES = 55
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

def get_zones_fcc(contcar_path):
    cell, positions_cart, atom_symbols = parse_poscar(contcar_path)
    h_indices = [i for i, s in enumerate(atom_symbols) if s == 'H']
    metal_indices = [i for i, s in enumerate(atom_symbols) if s != 'H']
    metal_pos = positions_cart[metal_indices]
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
    h_pos = positions_cart[h_indices[0]]
    top_idx = np.where(super_layer == 'top')[0]
    dist_top = np.linalg.norm(super_pos[top_idx] - h_pos, axis=1)
    top_sorted = top_idx[np.argsort(dist_top)]
    zone1_indices = top_sorted[:3]
    zone1_syms = [super_sym[i] for i in zone1_indices]
    zone1_pos = super_pos[zone1_indices]
    hollow_center = zone1_pos.mean(axis=0)
    top_excl = top_sorted[3:]
    dist_hollow_excl = np.linalg.norm(super_pos[top_excl] - hollow_center, axis=1)
    top_excl_sorted = top_excl[np.argsort(dist_hollow_excl)]
    zone4_indices = top_excl_sorted[:3]
    zone4_syms = [super_sym[i] for i in zone4_indices]
    zone2_indices = top_excl_sorted[3:9]
    zone2_syms = [super_sym[i] for i in zone2_indices]
    second_idx = np.where(super_layer == 'second')[0]
    dist_second = np.linalg.norm(super_pos[second_idx] - hollow_center, axis=1)
    second_sorted = second_idx[np.argsort(dist_second)]
    zone5_indices = second_sorted[:3]
    zone5_syms = [super_sym[i] for i in zone5_indices]
    zone3_indices = second_sorted[3:6]
    zone3_syms = [super_sym[i] for i in zone3_indices]
    return (zone1_syms, zone2_syms, zone3_syms, zone4_syms, zone5_syms)

def build_feature_vector(zone1_syms, zone2_syms, zone3_syms, zone4_syms, zone5_syms):
    key = tuple(sorted(zone1_syms, key=lambda s: ELEMENTS.index(s)))
    vec1 = [0.0] * len(ZONE1_COMBOS)
    vec1[ZONE1_COMBO_INDEX[key]] = 1.0
    vec_rest = []
    for zlist in [zone2_syms, zone3_syms, zone4_syms, zone5_syms]:
        for el in ELEMENTS:
            vec_rest.append(float(zlist.count(el)))
    return vec1 + vec_rest

def feature_vector(atoms):
    return build_feature_vector(*get_zones_fcc(atoms))

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
