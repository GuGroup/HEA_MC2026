#!/usr/bin/env python3
"""Evaluate packaged model without fitting; reproduce the original parity style.
All CSV test records are evaluated. Full-test and historical-plot metrics are separate.
Run: python evaluate_model.py [--output-dir PATH]
"""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def metrics(df):
    error = df["predicted_delE_H (eV)"] - df["DFT_delE_H (eV)"]
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    ss_res = float(np.sum(error ** 2))
    centered = df["DFT_delE_H (eV)"] - df["DFT_delE_H (eV)"].mean()
    r2 = 1.0 - ss_res / float(np.sum(centered ** 2))
    return mae, rmse, r2

def draw_parity_panel(ax, train_df, test_df, panel_label=None):
    train_metrics = metrics(train_df)
    test_metrics = metrics(test_df)
    ax.scatter(
        train_df["DFT_delE_H (eV)"],
        train_df["predicted_delE_H (eV)"],
        s=24, alpha=0.38, color="#4C78A8", edgecolors="none",
        label=f"Training set (N={len(train_df)})",
    )
    ax.scatter(
        test_df["DFT_delE_H (eV)"],
        test_df["predicted_delE_H (eV)"],
        s=80, alpha=0.95, color="red", marker="x", linewidths=1.8,
        label=f"Test set (N={len(test_df)})",
    )
    values = np.concatenate([
        train_df["DFT_delE_H (eV)"].to_numpy(),
        train_df["predicted_delE_H (eV)"].to_numpy(),
        test_df["DFT_delE_H (eV)"].to_numpy(),
        test_df["predicted_delE_H (eV)"].to_numpy(),
    ])
    margin = max(0.08, 0.06 * (values.max() - values.min()))
    lower, upper = values.min() - margin, values.max() + margin
    ax.plot([lower, upper], [lower, upper], color="black", lw=1.2, label="Ideal")
    ax.plot(
        [lower, upper], [lower + 0.1, upper + 0.1],
        "k--", lw=0.8, alpha=0.45, label=r"$\pm$0.1 eV",
    )
    ax.plot([lower, upper], [lower - 0.1, upper - 0.1], "k--", lw=0.8, alpha=0.45)
    ax.set(xlim=(lower, upper), ylim=(lower, upper))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(
        r"DFT $\Delta E_{\mathrm{H}}$ (eV)", fontsize=20, fontweight="bold"
    )
    ax.set_ylabel(
        r"Predicted $\Delta E_{\mathrm{H}}$ (eV)", fontsize=20, fontweight="bold"
    )
    ax.tick_params(axis="both", labelsize=20)
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    ax.legend(
        frameon=True, loc="lower right",
        prop={"size": 20, "weight": "bold"},
    )
    ax.text(
        0.03, 0.97,
        f"Training: RMSE={train_metrics[1]:.3f} eV, MAE={train_metrics[0]:.3f} eV\n"
        f"Test: RMSE={test_metrics[1]:.3f} eV, MAE={test_metrics[0]:.3f} eV",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=18, fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
    )
    if panel_label is not None:
        ax.text(
            -0.13, 1.015, panel_label,
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=22, fontweight="bold",
        )
    return train_metrics, test_metrics

def predict(data_path, trainer, model):
    data, X, y = trainer.load_dataset(data_path)
    coef = np.asarray(model['coef'], dtype=float)
    if model['elements'] != trainer.ELEMENTS or coef.shape != (X.shape[1],):
        raise ValueError('Model feature layout mismatch')
    pred = X @ coef
    return pd.DataFrame({'index': [r['index'] for r in data['records']],
                         'DFT_delE_H (eV)': y, 'predicted_delE_H (eV)': pred})


def select_original_plot(test, selection):
    if selection == 'absolute_error_le_0.1':
        keep = (test['predicted_delE_H (eV)'] - test['DFT_delE_H (eV)']).abs() <= 0.1
        return test.loc[keep].copy(), test.loc[~keep].copy()
    if selection == 'exclude_largest_3_absolute_errors':
        ranked = test.assign(absolute_error_eV=(test['predicted_delE_H (eV)'] - test['DFT_delE_H (eV)']).abs())
        ranked = ranked.sort_values(['absolute_error_eV', 'index'], ascending=[False, True], kind='mergesort')
        excluded = ranked.head(3)
        return test.drop(index=excluded.index).reset_index(drop=True), excluded
    if selection != 'all':
        raise ValueError('Unknown original plotting selection')
    return test.copy(), test.iloc[0:0].copy()


def metric_dict(frame):
    mae, rmse, r2 = metrics(frame)
    return {'n': len(frame), 'mae_eV': mae, 'rmse_eV': rmse, 'r2': r2}


def draw_one(train, test, path, label=None):
    fig, ax = plt.subplots(figsize=(9, 9))
    draw_parity_panel(ax, train, test, label)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=here / 'results')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    training = here.parent / 'training'
    spec = importlib.util.spec_from_file_location('site_trainer', training / 'train_model.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    model = json.loads((training / 'model.json').read_text())
    train = predict(training / 'training_data.json', trainer, model)
    test_all = predict(here / 'test_data.json', trainer, model)
    data_info = json.loads((here / 'test_data.json').read_text())
    policy = data_info['original_plot_selection']
    test_plot, excluded = select_original_plot(test_all, policy['rule'])
    for name, frame in [('train_predictions', train), ('test_predictions_all', test_all),
                        ('test_predictions_original_plot', test_plot), ('test_excluded_from_original_plot', excluded)]:
        frame.to_csv(out / (name + '.csv'), index=False)
    draw_one(train, test_plot, out / 'parity_original.png', policy.get('individual_panel_label'))
    if len(excluded):
        draw_one(train, test_all, out / 'parity_all_test.png')
    result = {'site': data_info['site'], 'training': metric_dict(train),
              'test_all': metric_dict(test_all), 'test_original_plot': metric_dict(test_plot),
              'original_plot_selection': policy['rule'],
              'excluded_from_original_plot_indices': [int(i) for i in excluded['index']],
              'note': 'Error-filtered subsets reproduce historical plots; use test_all for complete test-set performance.'}
    (out / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
