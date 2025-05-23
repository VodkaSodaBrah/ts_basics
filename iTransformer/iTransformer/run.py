import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_metrics(results_dir):
    npy_path = os.path.join(results_dir, 'metrics.npy')
    csv_path = os.path.join(results_dir, 'metrics.csv')
    if os.path.exists(npy_path):
        return np.load(npy_path, allow_pickle=True).item()
    elif os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        return df.to_dict(orient='list')
    else:
        raise FileNotFoundError(f"No metrics file found in {results_dir}")

def plot_metrics(metrics, setting, save_dir):
    epochs = range(1, len(metrics['train_loss']) + 1)
    plt.figure()
    plt.plot(epochs, metrics['train_loss'], label='Train Loss')
    plt.plot(epochs, metrics['val_loss'], label='Validation Loss')
    plt.title(f"Training and Validation Loss - {setting}")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{setting}_loss.png'))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--results_base', type=str, default='./iTransformer/results', help='Base directory for results')
    parser.add_argument('--setting', type=str, default=None, help='Specific setting to plot')
    parser.add_argument('--all', action='store_true', help='Plot all settings')
    args = parser.parse_args()

    results_base = args.results_base
    if not os.path.isdir(results_base) and os.path.isdir('./results'):
        results_base = './results'
        print(f"Warning: '{args.results_base}' not found, falling back to './results'")
    settings = sorted(os.listdir(results_base))
    settings = [
        s for s in settings
        if os.path.isdir(os.path.join(results_base, s))
           and (os.path.exists(os.path.join(results_base, s, 'metrics.npy')) 
                or os.path.exists(os.path.join(results_base, s, 'metrics.csv')))
    ]

    if args.all:
        selected_settings = settings
    elif args.setting is not None:
        if args.setting in settings:
            selected_settings = [args.setting]
        else:
            print(f"Warning: Setting {args.setting} not found or missing metrics file.")
            selected_settings = []
    else:
        if len(settings) == 0:
            print("No valid settings found with metrics files.")
            selected_settings = []
        else:
            selected_settings = [settings[-1]]

    for setting in selected_settings:
        results_dir = os.path.join(results_base, setting)
        try:
            metrics = load_metrics(results_dir)
            plot_metrics(metrics, setting, results_base)
            print(f"Plotted metrics for {setting}")
        except FileNotFoundError as e:
            print(e)