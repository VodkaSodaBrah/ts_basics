# plot_metrics.py
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ─── adjust these ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Plot iTransformer metrics")
parser.add_argument(
    "--setting",
    type=str,
    required=False,
    default=None,
    help="(optional) Name of the setting folder under checkpoints/bnb_exp and results; if omitted, uses the latest."
)
parser.add_argument(
    "--all",
    action="store_true",
    help="Plot metrics for all available settings"
)
args = parser.parse_args()
setting = args.setting
all_flag = args.all
base_dir = os.path.dirname(os.path.abspath(__file__))
# locate checkpoints/bnb_exp directory (outer or inner)
outer_ckpt = os.path.join(base_dir, "checkpoints", "bnb_exp")
inner_ckpt = os.path.join(base_dir, "iTransformer", "checkpoints", "bnb_exp")
if os.path.isdir(outer_ckpt):
    exp_dir = outer_ckpt
elif os.path.isdir(inner_ckpt):
    exp_dir = inner_ckpt
else:
    raise FileNotFoundError(f"No checkpoints/bnb_exp directory found in {base_dir} or its iTransformer subfolder")
if setting is None:
    # get all subdirs
    candidates = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    if not candidates:
        raise ValueError(f"No experiment subfolders found in {exp_dir}")
    print("Available settings:", candidates, "(use --all to plot all)")
    if all_flag:
        settings = sorted(candidates)
    else:
        settings = [sorted(candidates)[-1]]
        print(f"No --setting provided, defaulting to latest: {settings[0]}")
else:
    settings = [setting]
# ────────────────────────────────────────────────────────────────────────────────

for setting in settings:
    checkpoint_dir = os.path.join(exp_dir, setting)
    # 1) load per‐epoch losses (always present in checkpoint_dir)
    metrics_df = pd.read_csv(os.path.join(checkpoint_dir, "metrics.csv"))
    # define output directory for plots (always use checkpoint_dir)
    out_dir = checkpoint_dir

    # optionally locate a separate results folder for final metrics and predictions
    nested = os.path.join(base_dir, "iTransformer", "results", setting)
    flat   = os.path.join(base_dir, "results", setting)
    if os.path.isdir(nested):
        results_dir = nested
    elif os.path.isdir(flat):
        results_dir = flat
    else:
        results_dir = None
        if all_flag:
            print(f"Warning: no results folder for {setting}, skipping test metrics/preds")

    # plot train/val/test loss over epochs
    plt.figure(figsize=(8,4))
    epochs = metrics_df['epoch'].tolist()
    plt.plot(epochs, metrics_df['train_loss'], label='Train', marker='o')
    plt.plot(epochs, metrics_df['val_loss'],   label='Validation', marker='s')
    plt.plot(epochs, metrics_df['test_loss'],  label='Test', marker='^')
    plt.xticks(epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('iTransformer Loss Curve')
    plt.grid(True, linestyle='-', alpha=0.3)
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter())  # use plain numbers
    # increase line width and marker size
    for line in ax.get_lines():
        line.set_linewidth(2)
        line.set_markersize(8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{setting}_loss_curve.png"), dpi=150)
    # plt.show()

    if results_dir is not None:
        # 2) load final test metrics array [MAE, MSE, RMSE, MAPE, MSPE]
        try:
            test_metrics = np.load(f"{results_dir}/metrics.npy")
            metric_names = ['MAE','MSE','RMSE','MAPE','MSPE']
        except FileNotFoundError:
            # fallback to last epoch test_loss from metrics_df
            test_metrics = np.array([metrics_df['test_loss'].iloc[-1]])
            metric_names = ['Test Loss']
            print("metrics.npy not found, using last test_loss from metrics.csv")

        # plot final test metrics as bar chart
        plt.figure(figsize=(6,4))
        plt.bar(metric_names, test_metrics)
        plt.ylabel('Value')
        plt.title('Final Test Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{setting}_test_metrics.png"), dpi=150)
        # plt.show()

        # plot sample predictions and error distribution
        pred_file = os.path.join(results_dir, 'pred.npy')
        true_file = os.path.join(results_dir, 'true.npy')
        if os.path.isfile(pred_file) and os.path.isfile(true_file):
            preds = np.load(pred_file)
            trues = np.load(true_file)
            # reshape if needed
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            # sample 0
            plt.figure(figsize=(8,4))
            plt.plot(trues[0,:,0], label='True')
            plt.plot(preds[0,:,0], '--', label='Pred')
            plt.xlabel('Time-step')
            plt.ylabel('Scaled Value')
            plt.title('Sample Prediction vs True')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{setting}_pred_vs_true.png"), dpi=150)
            # plt.show()
            # error histogram
            errors = (preds - trues).ravel()
            plt.figure(figsize=(6,4))
            plt.hist(errors, bins=50, alpha=0.7)
            plt.xlabel('Error')
            plt.ylabel('Count')
            plt.title('Prediction Error Distribution')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{setting}_error_histogram.png"), dpi=150)
            # plt.show()
        else:
            print("pred.npy / true.npy not found in", results_dir)