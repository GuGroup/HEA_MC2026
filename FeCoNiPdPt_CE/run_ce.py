#!/usr/bin/env python3
"""Portable entry point using the bundled, unmodified fcc-ce source."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / '.cache' / 'numba'))
import numpy as np
import pandas as pd
from fcc_ce.config import load_config
from fcc_ce.data import discover_structures, match_targets, read_targets
from fcc_ce.lattice import map_structure
from fcc_ce.model import CEModel, train_model, save_training_outputs, _metric_block
from fcc_ce.topology import build_topology, topology_signature


def config_for(facet):
    folder = ROOT / ('fcc' + facet)
    config = load_config(folder / 'train.yaml')
    for key in ['structures', 'targets']:
        config['data'][key] = str((folder / config['data'][key]).resolve())
    return config


def train(facet):
    output = ROOT / 'results' / ('fcc' + facet) / 'retrained'
    model, frame, metrics = train_model(config_for(facet), progress=lambda s: print(s, flush=True))
    save_training_outputs(model, frame, metrics, output)
    reference = ROOT / ('fcc' + facet) / 'reference'
    previous = pd.read_csv(reference / 'training_predictions.csv', dtype={'identifier': str})
    original = CEModel.load(reference / 'model')
    if frame['identifier'].astype(str).tolist() != previous['identifier'].tolist():
        raise AssertionError('Training record order changed')
    if frame['split'].tolist() != previous['split'].tolist():
        raise AssertionError('Fit/holdout membership changed')
    report = {'facet': facet, 'same_sample_order': True, 'same_holdout_split': True,
              'selected_alpha': metrics['selected_alpha'],
              'reference_alpha': original.metadata['metrics']['selected_alpha'],
              'coefficient_max_abs_difference': float(np.max(np.abs(model.coefficients - original.coefficients)))}
    assert report['selected_alpha'] == report['reference_alpha']
    for column in ['target', 'prediction', 'validation_prediction']:
        delta = float(np.max(np.abs(frame[column].to_numpy() - previous[column].to_numpy())))
        report[column + '_max_abs_difference'] = delta
        np.testing.assert_allclose(frame[column], previous[column], rtol=0, atol=1e-8)
    np.testing.assert_allclose(model.coefficients, original.coefficients, rtol=0, atol=1e-7)
    (output / 'reproduction_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print('REPRODUCTION', json.dumps(report), flush=True)


def evaluate(facet, use_retrained=False):
    config = config_for(facet)
    reference = ROOT / ('fcc' + facet) / 'reference'
    source = ROOT / 'results' / ('fcc' + facet) / 'retrained' if use_retrained else reference
    output = ROOT / 'results' / ('fcc' + facet) / ('evaluation_retrained' if use_retrained else 'evaluation_reference')
    output.mkdir(parents=True, exist_ok=True)
    model = CEModel.load(source / 'model')
    saved = pd.read_csv(source / 'training_predictions.csv', dtype={'identifier': str})
    records, missing = match_targets(read_targets(config['data']['targets']),
        discover_structures(config['data']['structures'], config['data']['extensions']), strict=True)
    assert not missing and [r.identifier for r in records] == saved['identifier'].tolist()
    cache, rows = {}, []
    for i, record in enumerate(records, 1):
        mapped = map_structure(record.path, model.mapper_config)
        signature = topology_signature(mapped)
        if signature not in cache:
            cache[signature] = build_topology(mapped, model.cluster_spec)
        prediction = model.predict_mapped(mapped, cache[signature])
        rows.append({'identifier': record.identifier, 'target': record.target, **prediction})
        if i % 250 == 0 or i == len(records):
            print(f'fcc{facet}: evaluated {i}/{len(records)}', flush=True)
    fresh = pd.DataFrame(rows)
    np.testing.assert_allclose(fresh['prediction'], saved['prediction'], rtol=0, atol=1e-8)
    np.testing.assert_allclose(fresh['target'], saved['target'], rtol=0, atol=1e-12)
    fresh.to_csv(output / 'saved_model_predictions.csv', index=False)
    y = saved['target'].to_numpy()
    labels = saved['kind'].to_numpy()
    holdout = saved['split'].eq('holdout').to_numpy()
    validation = saved['validation_prediction'].to_numpy()
    result = {'full_refit': _metric_block(y, fresh['prediction'].to_numpy(), labels),
              'training_split': _metric_block(y[~holdout], validation[~holdout], labels[~holdout]),
              'holdout': _metric_block(y[holdout], validation[holdout], labels[holdout]),
              'saved_model_prediction_max_abs_difference': float(np.max(np.abs(fresh['prediction'] - saved['prediction']))),
              'holdout_source': str((source / 'training_predictions.csv').relative_to(ROOT)),
              'note': 'Saved model is the full-data refit. Holdout metrics use validation_prediction from the fit-only model; run train to regenerate them.'}
    expected = json.loads((source / 'metrics.json').read_text())
    for section in ['full_refit', 'training_split', 'holdout']:
        for key in ['mae', 'rmse', 'r2']:
            np.testing.assert_allclose(result[section][key], expected[section][key], rtol=0, atol=1e-10)
        assert result[section]['n'] == expected[section]['n']
    (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    draw_parity(facet, saved, output)
    print('EVALUATED', facet, json.dumps({k: result[k] for k in ['training_split','holdout','full_refit']}), flush=True)


def draw_parity(facet, saved, output):
    """Plot only held-out predictions from the model fitted on the fit subset."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    part = saved.loc[saved['split'].eq('holdout')].copy()
    if part.empty:
        raise ValueError('No holdout records available for the parity plot')
    y = part['target'].to_numpy()
    prediction = part['validation_prediction'].to_numpy()
    mae = mean_absolute_error(y, prediction) * 1000
    rmse = np.sqrt(mean_squared_error(y, prediction)) * 1000
    r2 = r2_score(y, prediction)
    alloy = 'FeCoNiPdPt'
    fig, ax = plt.subplots(figsize=(5.4, 5.4), layout='constrained')
    for kind, group in part.groupby('kind', sort=True):
        is_bulk = str(kind).lower() == 'bulk'
        ax.scatter(group['target'], group['validation_prediction'],
                   s=9, alpha=0.85, linewidths=0,
                   color='#4C91CF' if is_bulk else '#FF7F0E',
                   marker='o' if is_bulk else '^',
                   label='Bulk' if is_bulk else f'{alloy} ({facet}) slab')
    values = np.concatenate([y, prediction])
    span = float(np.ptp(values))
    margin = max(span * 0.05, 0.005)
    lo, hi = values.min() - margin, values.max() + margin
    ax.plot([lo, hi], [lo, hi], color='black', linestyle='--', lw=0.7)
    ax.set(xlim=(lo, hi), ylim=(lo, hi),
           xlabel='DFT-calculated formation energy (eV/atom)',
           ylabel='CE-predicted formation energy (eV/atom)')
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(direction='in', top=True, right=True)
    annotation = (f'$n$ = {len(part)}\n'
                  f'MAE = {mae:.3f} meV/atom\n'
                  f'RMSE = {rmse:.3f} meV/atom\n'
                  f'$R^2$ = {r2:.5f}')
    ax.text(0.04, 0.88, annotation, transform=ax.transAxes,
            va='top', ha='left', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.8', lw=0.8))
    ax.legend(loc='lower right', frameon=False, fontsize=10)
    fig.savefig(output / 'parity.png', dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['train', 'evaluate', 'predict'])
    p.add_argument('--facet', choices=['111', '100', '110', 'all'], default='all')
    p.add_argument('--use-retrained', action='store_true')
    p.add_argument('--inputs', nargs='+')
    p.add_argument('--output', default='new_predictions.csv')
    args = p.parse_args()
    facets = ['111','100','110'] if args.facet == 'all' else [args.facet]
    if args.command == 'predict':
        if len(facets) != 1 or not args.inputs:
            p.error('predict requires one --facet and --inputs')
        from fcc_ce.cli import main as cli
        model = ROOT / ('fcc' + args.facet) / 'reference/model'
        if args.use_retrained:
            model = ROOT / 'results' / ('fcc' + args.facet) / 'retrained/model'
        raise SystemExit(cli(['predict','--model',str(model),'--inputs',*args.inputs,'--output',args.output]))
    for facet in facets:
        print(args.command.upper(), 'fcc' + facet, flush=True)
        train(facet) if args.command == 'train' else evaluate(facet, args.use_retrained)


if __name__ == '__main__':
    main()
