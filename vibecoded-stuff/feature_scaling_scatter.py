"""Plot feature values before and after standard scaling."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_FILE = Path(r"D:\ThesisData\front_trees\1020\1715\data.csv")
OUTPUT_DIR = Path(__file__).parent

data = pd.read_csv(DATA_FILE)
features = data.select_dtypes(include='number').columns.tolist()
values = data[features].to_numpy()
scaled_values = StandardScaler().fit_transform(values)

# The same offsets, color mapping, and point jitter are used in both figures.
rng = np.random.default_rng(42)
y_jitter = rng.uniform(-0.25, 0.25, size=values.shape)
colors = plt.get_cmap('tab20').colors


def scatter_features(matrix, title, filename):
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=180)
    for i, feature in enumerate(features):
        ax.scatter(
            matrix[:, i], i + y_jitter[:, i],
            label=feature, color=colors[i % len(colors)],
            s=7, alpha=0.35, edgecolors='none', rasterized=True,
        )
    ax.set_title(title, fontsize=16, pad=15)
    ax.set_xlabel('Feature value', fontsize=12)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_yticks(range(len(features)), features)
    ax.set_ylim(-0.65, len(features) - 0.35)
    ax.grid(axis='x', alpha=0.22)
    ax.set_axisbelow(True)
    # ax.legend(title='Features', loc='center left', bbox_to_anchor=(1.02, 0.5),
    #           markerscale=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches='tight')
    plt.close(fig)


scatter_features(values, 'Feature Values before Standard Scaling', 'before_standard_scaling.png')
scatter_features(scaled_values, 'Feature Values after Standard Scaling', 'after_standard_scaling.png')
