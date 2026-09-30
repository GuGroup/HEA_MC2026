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
MODEL_METADATA = {'description': '*H (fcc(110) bridge) adsorption energy linear model for FeCoNiPdPt HEA', 'elements': ['Fe', 'Co', 'Ni', 'Pd', 'Pt'], 'binding_site': 'fcc110_bridge', 'zone_sizes': [2, 6, 2, 4], 'n_parameters': 40}
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

def already_seen(pos, seen_list, tol=0.5):
    return any((np.linalg.norm(pos - sp) < tol for sp in seen_list))

def unique_by_position(indices, positions, tol=0.5):
    unique = []
    seen = []
    for idx in indices:
        if not already_seen(positions[idx], seen, tol=tol):
            unique.append(idx)
            seen.append(positions[idx])
    return unique

def get_zones_fcc110_bridge(poscar_path, layer_tol=0.5, repeat_range=range(-2, 3)):
    """Return zone symbols for fcc(110) bridge H adsorption.

    zone1: top-layer metal atoms closest to H, 2 atoms.
    zone2: top-layer nearest neighbors of each zone1 atom, 3 atoms per zone1
           atom, excluding only the selected zone1 atom positions, total 6 atoms.
    zone3: second-layer atoms bonded to both zone1 atoms, 2 atoms.
    zone4: second-layer atoms bonded to one zone1 atom, excluding zone3 atoms,
           2 atoms per zone1 atom, total 4 atoms.
    """
    cell, positions_cart, atom_symbols = parse_poscar(poscar_path)
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
    super_pos = np.array(super_pos, dtype=float)
    super_layer = np.array(super_layer)
    h_pos = positions_cart[h_indices[0]]
    top_idx = np.where(super_layer == 'top')[0]
    second_idx = np.where(super_layer == 'second')[0]
    top_by_h = top_idx[np.argsort(np.linalg.norm(super_pos[top_idx] - h_pos, axis=1))]
    zone1_idx = unique_by_position(top_by_h, super_pos, tol=layer_tol)[:2]
    if len(zone1_idx) < 2:
        raise ValueError(f'zone1 has only {len(zone1_idx)} atoms')
    zone1_pos = [super_pos[i] for i in zone1_idx]
    zone1_syms = [super_sym[i] for i in zone1_idx]
    zone1_idx_set = set(zone1_idx)
    top_excl_idx = np.array([i for i in top_idx if i not in zone1_idx_set])
    zone2_idx = []
    zone2_pos = []
    for z1i in zone1_idx:
        by_z1 = top_excl_idx[np.argsort(np.linalg.norm(super_pos[top_excl_idx] - super_pos[z1i], axis=1))]
        added = 0
        for ni in by_z1:
            if already_seen(super_pos[ni], zone2_pos, tol=layer_tol):
                continue
            zone2_idx.append(ni)
            zone2_pos.append(super_pos[ni])
            added += 1
            if added == 3:
                break
        if added < 3:
            raise ValueError(f'zone2 added only {added} atoms for one zone1 atom')
    if len(zone2_idx) < 6:
        raise ValueError(f'zone2 has only {len(zone2_idx)} atoms')
    zone2_syms = [super_sym[i] for i in zone2_idx[:6]]

    def shared_second_key(idx):
        dists = np.linalg.norm(super_pos[idx] - np.array(zone1_pos), axis=1)
        return (float(np.max(dists)), float(np.sum(dists)))
    second_by_pair = sorted(second_idx, key=shared_second_key)
    zone3_idx = unique_by_position(second_by_pair, super_pos, tol=layer_tol)[:2]
    if len(zone3_idx) < 2:
        raise ValueError(f'zone3 has only {len(zone3_idx)} atoms')
    zone3_pos = [super_pos[i] for i in zone3_idx]
    zone3_syms = [super_sym[i] for i in zone3_idx]
    zone4_idx = []
    zone4_pos = []
    for z1i in zone1_idx:
        by_z1 = second_idx[np.argsort(np.linalg.norm(super_pos[second_idx] - super_pos[z1i], axis=1))]
        added = 0
        for ni in by_z1:
            if already_seen(super_pos[ni], zone3_pos, tol=layer_tol):
                continue
            if already_seen(super_pos[ni], zone4_pos, tol=layer_tol):
                continue
            zone4_idx.append(ni)
            zone4_pos.append(super_pos[ni])
            added += 1
            if added == 2:
                break
        if added < 2:
            raise ValueError(f'zone4 added only {added} atoms for one zone1 atom')
    if len(zone4_idx) < 4:
        raise ValueError(f'zone4 has only {len(zone4_idx)} atoms')
    zone4_syms = [super_sym[i] for i in zone4_idx[:4]]
    return (zone1_syms, zone2_syms, zone3_syms, zone4_syms)

def build_feature_vector(zone1_syms, zone2_syms, zone3_syms, zone4_syms):
    key = tuple(sorted(zone1_syms, key=lambda s: ELEMENTS.index(s)))
    vec1 = [0.0] * len(ZONE1_COMBOS)
    vec1[ZONE1_COMBO_INDEX[key]] = 1.0
    vec2 = []
    for el in ELEMENTS:
        vec2.append(float(zone2_syms.count(el)))
    key3 = tuple(sorted(zone3_syms, key=lambda s: ELEMENTS.index(s)))
    vec3 = [0.0] * len(ZONE3_COMBOS)
    vec3[ZONE3_COMBO_INDEX[key3]] = 1.0
    vec4 = []
    for el in ELEMENTS:
        vec4.append(float(zone4_syms.count(el)))
    return vec1 + vec2 + vec3 + vec4

def feature_vector(atoms):
    return build_feature_vector(*get_zones_fcc110_bridge(atoms))

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
