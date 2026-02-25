import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
plt.rcParams['font.weight'] = 'bold'
import seaborn as sns
import os
# configs
sns.set_theme(style="ticks")

import numpy as np
import json

PATH_PLOTS = "./docs/local_data/visualizations/"

results_fpr95_failurenet = {
    1: 4.29, 2: 4.06, 3: 4.36, 4: 5.35, 5: 4.83, 6: 5.27, 7: 6.07,
}

results_fpr95_failurenet_features = {
    16: 5.47, 32: 5.83, 64: 4.06, 96: 4.35, 128: 4.27, 
}

results_fpr95_failurenet_data_efficiency = {
    'surgi4k':     {20: 4.22, 40: 4.54, 60: 5.68, 80: 4.69, 100: 4.06},
    'hyperkvasir': {20: 36.88, 40: 6.53, 60: 6.18, 80: 5.52, 100: 7.33},
}

# Ablation study on number of blocks for Failure Network

fig, ax = plt.subplots(figsize=(6.4, 3.6))

plt.plot(results_fpr95_failurenet.keys(), results_fpr95_failurenet.values(), linestyle='-', marker='s', color="darkviolet", alpha=0.4, markersize=14, linewidth=5,
         markeredgecolor='k', zorder=3)

plt.legend(labels=[r"Surgi-2$\times$"], loc='upper left', prop={'weight': 'bold', 'size': 18}, framealpha=1)

plt.plot(results_fpr95_failurenet.keys(), results_fpr95_failurenet.values(), linestyle='none', marker='s', color="darkviolet", alpha=1, markersize=14, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)

plt.plot([2, 2], [-5, 4], linestyle='--', marker='none', color="red", linewidth=3, alpha=0.4, zorder=2)

plt.xlabel(r"Number of blocks", fontsize=25, weight="bold")
plt.ylabel(r'FPR95 (%)', fontsize=25, weight="bold")
plt.tick_params(axis='both', which='major', labelsize=23)
ax = plt.gca()
plt.yticks([3, 4, 5, 6],[3, 4, 5, 6])
plt.xticks([1, 2, 3, 4, 5, 6, 7],[1, 2, 3, 4, 5, 6, 7])
ax = plt.gca().axis()
plt.gca().axis((0.5, 7.5, 3, 6.5))
ax = plt.gca()
for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
plt.tight_layout()
plt.grid(axis="both", linestyle='--', linewidth=2, zorder=-5)

if not os.path.exists(PATH_PLOTS +  "ablation_studies/"):
        os.makedirs(PATH_PLOTS +  "ablation_studies/")
path = PATH_PLOTS +  "ablation_studies/number_blocks.png"
fig.set_size_inches(6.5, 3.5)
plt.savefig(path, bbox_inches='tight')
plt.close()

# Ablation study on number features in each block

fig, ax = plt.subplots(figsize=(6.4, 3.6))

plt.plot(results_fpr95_failurenet_features.keys(), results_fpr95_failurenet_features.values(), linestyle='-', marker='s', color="darkviolet", alpha=0.4, markersize=14, linewidth=5,
         markeredgecolor='k', zorder=3)

plt.legend(labels=[r"Surgi-2$\times$"], loc='upper right', prop={'weight': 'bold', 'size': 18}, framealpha=1)

plt.plot(results_fpr95_failurenet_features.keys(), results_fpr95_failurenet_features.values(), linestyle='none', marker='s', color="darkviolet", alpha=1, markersize=14, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)

plt.plot([64, 64], [-5, 4], linestyle='--', marker='none', color="red", linewidth=3, alpha=0.4, zorder=2)

plt.xlabel(r"Number of features", fontsize=25, weight="bold")
plt.ylabel(r'FPR95 (%)', fontsize=25, weight="bold")
plt.tick_params(axis='both', which='major', labelsize=23)
ax = plt.gca()
plt.yticks([3, 4, 5, 6],[3, 4, 5, 6])
plt.xticks([16, 32, 64, 96, 128],[16, 32, 64, 96, 128])
ax = plt.gca().axis()
plt.gca().axis((10, 134, 3, 6.5))
ax = plt.gca()
for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
plt.tight_layout()
plt.grid(axis="both", linestyle='--', linewidth=2, zorder=-5)

if not os.path.exists(PATH_PLOTS +  "ablation_studies/"):
        os.makedirs(PATH_PLOTS +  "ablation_studies/")
path = PATH_PLOTS +  "ablation_studies/number_features.png"
fig.set_size_inches(6.5, 3.5)
plt.savefig(path, bbox_inches='tight')
plt.close()

# Ablation study on training-data efficiency

fig, ax = plt.subplots(figsize=(6.4, 3.6))

plt.plot(results_fpr95_failurenet_data_efficiency['surgi4k'].keys(), results_fpr95_failurenet_data_efficiency['surgi4k'].values(), linestyle='-', marker='s', color="darkviolet", alpha=0.4, markersize=14, linewidth=5,
         markeredgecolor='k', zorder=3)
plt.plot(results_fpr95_failurenet_data_efficiency['hyperkvasir'].keys(), results_fpr95_failurenet_data_efficiency['hyperkvasir'].values(), linestyle='-', marker='s', color="thistle", alpha=0.4, markersize=14, linewidth=5,
         markeredgecolor='k', zorder=3)

plt.legend(labels=[r"Surgi-2$\times$", "HKvasir"], loc='upper right', prop={'weight': 'bold', 'size': 18}, framealpha=1)

plt.plot(results_fpr95_failurenet_data_efficiency['surgi4k'].keys(), results_fpr95_failurenet_data_efficiency['surgi4k'].values(), linestyle='none', marker='s', color="darkviolet", alpha=1, markersize=14, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)
plt.plot(results_fpr95_failurenet_data_efficiency['hyperkvasir'].keys(), results_fpr95_failurenet_data_efficiency['hyperkvasir'].values(), linestyle='none', marker='s', color="thistle", alpha=1, markersize=14, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)

plt.xlabel(r"Training data (%)", fontsize=25, weight="bold")
plt.ylabel(r'FPR95 (%)', fontsize=25, weight="bold")
plt.tick_params(axis='both', which='major', labelsize=23)
ax = plt.gca()
plt.yticks([5, 15, 25, 35],[5, 15, 25, 35])
plt.xticks([20, 40, 60, 80, 100],[20, 40, 60, 80, 100])
ax = plt.gca().axis()
plt.gca().axis((10, 110, 0, 40))
ax = plt.gca()
for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
plt.tight_layout()
plt.grid(axis="both", linestyle='--', linewidth=2, zorder=-5)

if not os.path.exists(PATH_PLOTS +  "ablation_studies/"):
        os.makedirs(PATH_PLOTS +  "ablation_studies/")
path = PATH_PLOTS +  "ablation_studies/data_efficiency.png"
fig.set_size_inches(6.5, 3.5)
plt.savefig(path, bbox_inches='tight')
plt.close()