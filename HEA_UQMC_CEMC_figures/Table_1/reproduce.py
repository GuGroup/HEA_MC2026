#!/usr/bin/env python3
"""Recompute Table 1 from paired UQMC scores; no simulation server is needed."""
import argparse
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
METRICS = ('tau', 'mse', 'crps')

def calculate(data_dir):
    metadata = json.loads((data_dir / 'provenance.json').read_text())
    rows = []
    checks = []
    for case in metadata['datasets']:
        path = data_dir / case['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != case['sha256']:
            raise ValueError(f'Input checksum mismatch: {path}')
        with gzip.open(path, 'rt', newline='') as stream:
            values = list(csv.DictReader(stream))
        trial = np.array([int(v['trial']) for v in values])
        if not np.array_equal(np.sort(trial), np.arange(10000)):
            raise ValueError(f'{case["dataset"]}: expected unique trial IDs 0..9999')
        c = np.array([[float(v['cemc_' + m]) for m in METRICS] for v in values])
        h = np.array([[float(v['homogeneous_' + m]) for m in METRICS] for v in values])
        if not np.isfinite(c).all() or not np.isfinite(h).all():
            raise ValueError('Scores must be finite; missing values cannot be silently dropped')
        delta = (c - h) * np.array([1., -1., -1.])
        lower, upper = np.quantile(delta, [.025, .975], axis=0, method='linear')
        result = {'dataset': case['dataset'], 'n_trials': len(trial)}
        for j, metric in enumerate(METRICS):
            result['p_' + metric] = float(np.mean(delta[:, j] > 0))
            result[metric + '_q025'] = float(lower[j])
            result[metric + '_q975'] = float(upper[j])
        result['p_all'] = float(np.mean(np.all(delta > 0, axis=1)))
        rows.append(result)
        checks.append({'dataset': case['dataset'], 'unique_paired_trials': len(trial),
                       'strictly_positive_counts': (delta > 0).sum(axis=0).tolist(),
                       'all_positive_count': int(np.all(delta > 0, axis=1).sum()),
                       'ties_per_metric': (delta == 0).sum(axis=0).tolist()})
    average = {'dataset': 'Average', 'n_trials': ''}
    for metric in (*METRICS, 'all'):
        average['p_' + metric] = float(np.mean([r['p_' + metric] for r in rows]))
    rows.append(average)
    return rows, checks

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    args = parser.parse_args()
    rows, checks = calculate(args.data_dir)
    reference = json.loads((ROOT / 'manuscript_reference.json').read_text())
    mismatches = []
    compared = 0
    for actual, expected in zip(rows, reference['rows'], strict=True):
        assert actual['dataset'] == expected['dataset']
        for key, value in expected.items():
            if key == 'dataset':
                continue
            compared += 1
            rounded = f'{actual[key]:.2f}'
            if rounded != value:
                mismatches.append({'dataset': actual['dataset'], 'field': key,
                                   'manuscript': value, 'recomputed': rounded,
                                   'unrounded': actual[key]})
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with (out / 'table1_full_precision.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    headers = ['Dataset', 'P(delta tau > 0) [2.5%, 97.5%]',
               'P(delta MSE > 0) [2.5%, 97.5%]',
               'P(delta CRPS > 0) [2.5%, 97.5%]', 'P(all three > 0)']
    formatted = []
    for row in rows:
        cells = [row['dataset']]
        for metric in METRICS:
            cell = f'{row["p_" + metric]:.2f}'
            if row['dataset'] != 'Average':
                cell += f' [{row[metric + "_q025"]:.2f}, {row[metric + "_q975"]:.2f}]'
            cells.append(cell)
        cells.append(f'{row["p_all"]:.2f}')
        formatted.append(cells)
    with (out / 'table1.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(headers); writer.writerows(formatted)
    note = ('Intervals are the 2.5th and 97.5th percentiles of paired trial differences, '
            'not confidence intervals for the improvement probabilities. '
            'Average gives equal weight to the 12 datasets. '
            'Values are recomputed from the supplied scores; manuscript discrepancies are listed below.')
    md = '# Table 1: CEMC improvement probabilities\n\n' + note + '\n\n'
    md += '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * 5) + '|\n'
    md += ''.join('| ' + ' | '.join(cells) + ' |\n' for cells in formatted)
    md += '\n## Comparison with the supplied manuscript\n\n'
    for item in mismatches:
        md += f'- {item["dataset"]}, `{item["field"]}`: manuscript {item["manuscript"]}; recomputed {item["recomputed"]}.\n'
    md += '\nAll improvement probabilities, including the Average row, match the manuscript at two decimal places.\n' if not any(x['field'].startswith('p_') for x in mismatches) else '\nSee validation.json for probability discrepancies.\n'
    (out / 'table1.md').write_text(md)
    table = '<tr>' + ''.join('<th>' + html.escape(v) + '</th>' for v in headers) + '</tr>'
    table += ''.join('<tr>' + ''.join('<td>' + html.escape(v) + '</td>' for v in cells) + '</tr>' for cells in formatted)
    (out / 'table1.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Table 1</title><style>body{font:15px Arial;margin:30px}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:10px}th{background:#eee}</style><h1>Table 1</h1><p>' + html.escape(note) + '</p><table>' + table + '</table><h2>Manuscript comparison</h2><pre>' + html.escape(json.dumps(mismatches, indent=2)) + '</pre></html>')
    validation = {'numpy_version': np.__version__, 'quantile_method': 'linear',
                  'comparison': 'strict delta > 0; equal trial weights; pair by trial ID',
                  'compared_numeric_cells': compared, 'matching_cells': compared - len(mismatches),
                  'mismatches': mismatches, 'datasets': checks}
    (out / 'validation.json').write_text(json.dumps(validation, indent=2))
    print(f'Wrote Table 1 to {out}: {compared - len(mismatches)}/{compared} manuscript values match at two decimals.')
    for item in mismatches:
        print(f'  {item["dataset"]} {item["field"]}: {item["manuscript"]} -> {item["recomputed"]}')

if __name__ == '__main__':
    main()
