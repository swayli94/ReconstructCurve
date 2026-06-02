import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from curve_reconstruction import order_curve_cw


OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

data_files = sorted(glob.glob('data/*.csv'))

for fpath in data_files:
    name = os.path.splitext(os.path.basename(fpath))[0]
    df = pd.read_csv(fpath, header=0)
    pts = df.values.astype(float)   # shape (N, 2): col0=x, col1=z

    try:
        order = order_curve_cw(pts)
        ordered = pts[order]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(name, fontsize=13)

        ax = axes[0]
        ax.scatter(pts[:, 0], pts[:, 1], s=8, color='gray', alpha=0.6, label='scattered pts')
        ax.set_title('Input scatter')
        # ax.set_aspect('equal')
        ax.legend(fontsize=8)

        ax = axes[1]
        closed = np.vstack([ordered, ordered[0]])   # close the loop for plotting
        ax.plot(closed[:, 0], closed[:, 1], '-o', markersize=3, linewidth=1.0,
                color='steelblue', label='reconstructed CW')
        ax.plot(ordered[0, 0], ordered[0, 1], 'r*', markersize=10, label='start (max x)')
        ax.set_title('Reconstructed curve')
        # ax.set_aspect('equal')
        ax.legend(fontsize=8)

        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f'{name}.png')
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'[OK]  {name} -> start point {ordered[0, 0]}, {ordered[0, 1]}')

    except Exception as e:
        print(f'[ERR] {name}: {e}')
