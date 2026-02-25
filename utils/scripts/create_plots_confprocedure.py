import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['legend.handlelength']=1.0
import seaborn as sns
import os
# configs
sns.set_theme(style="ticks")

import numpy as np
import json

PATH_PLOTS = "./docs/local_data/visualizations/"

results = {
    'surgi4k': {
        'size': {12: 00, 13: 00, 14: 0.2, 15: 0.8, 16: 2.0, 17: 3.2, 18: 5.0, 19: 7.8, 20: 15.6, 21: 45.6, 22: 84.3, 23: 97.9, 24: 100, 25: 100, 26: 100},
        'fnr':  {12: 100, 13: 100, 14: 78.1, 15: 42.3, 16: 16.2, 17: 7.3, 18: 2.8, 19: 1.0, 20: 0.3, 21: 0.08, 22: 0.02, 23: 00, 24: 00, 25: 00, 26: 00, },
    },
    'hkvasir': {
        'size': {12: 4.5, 13: 8.0, 14: 13.5, 15: 25.0, 16: 44.8, 17: 67.6, 18: 90.2, 19: 97.1, 20: 99.2, 21: 99.8, 22: 100, 23: 100, 24: 100, 25: 100, 26: 100},
        'fnr':  {12: 12.7, 13: 6.2, 14: 3.3, 15: 1.7, 16: 0.8, 17: 0.3, 18: 0.1, 19: 0.05, 20: 0.02, 21: 0.01, 22: 00, 23: 00, 24: 00, 25: 00, 26: 00, },
    },
}

results_ours = {
    'hkvasir': {
        'size': [96.4, 71.7, 40.0, 18.1, 10.0,  5.6,  4.0],
        'fnr':  [ 0.1, 0.3,  0.9,  2.5,  5.0, 10.2, 15.0],
    },
}


# Study on the average mask size and FNR given a target PSNR

fig, ax = plt.subplots(figsize=(6.4, 3.6))

plt.plot(results['surgi4k']['size'].keys(), results['surgi4k']['size'].values(), linestyle='-', marker='s', color="darkviolet", alpha=0.4, markersize=12, linewidth=5,
         markeredgecolor='k', zorder=3)
plt.plot(results['hkvasir']['size'].keys(), results['hkvasir']['size'].values(), linestyle='-', marker='s', color="thistle", alpha=0.4, markersize=12, linewidth=5,
         markeredgecolor='k', zorder=3)

plt.legend(labels=[r"Surgi-2$\times$", "HKvasir"], loc='lower right', prop={'weight': 'bold', 'size': 18}, framealpha=1)

plt.plot(results['surgi4k']['size'].keys(), results['surgi4k']['size'].values(), linestyle='none', marker='s', color="darkviolet", alpha=1, markersize=12, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)
plt.plot(results['hkvasir']['size'].keys(), results['hkvasir']['size'].values(), linestyle='none', marker='s', color="thistle", alpha=1, markersize=12, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)

plt.plot([22, 22], [-5, 102], linestyle='--', marker='none', color="red", linewidth=3, alpha=0.4, zorder=2)

plt.xlabel(r"Target $\alpha_{[1]}$ (PSNR)", fontsize=24, weight="bold")
plt.ylabel(r'Mask (%)', fontsize=24, weight="bold")
plt.tick_params(axis='both', which='major', labelsize=20)
ax = plt.gca()
plt.yticks([0, 20, 40, 60, 80, 100],[0, 20, 40, 60, 80, 100])
plt.xticks([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
           ["12", "", "14", "", "16", "", "18", "", "20", "", "22", "", "24", "", "26"])
ax = plt.gca().axis()
plt.gca().axis((11.5, 26.5, -5, 105))
ax = plt.gca()
for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
plt.tight_layout()
plt.grid(axis="both", linestyle='--', linewidth=2, zorder=-5)

if not os.path.exists(PATH_PLOTS +  "conf_procedure/"):
        os.makedirs(PATH_PLOTS +  "conf_procedure/")
path = PATH_PLOTS +  "conf_procedure/avg_mask_size.png"
fig.set_size_inches(6.5, 3.5)
plt.savefig(path, bbox_inches='tight')
plt.close()


# Study on the average mask size and FNR given a target PSNR

fig, ax = plt.subplots(figsize=(6.4, 3.6))

plt.plot(results['surgi4k']['size'].keys(), results['surgi4k']['fnr'].values(), linestyle='-', marker='s', color="darkviolet", alpha=0.4, markersize=12, linewidth=5,
         markeredgecolor='k', zorder=3)
plt.plot(results['hkvasir']['size'].keys(), results['hkvasir']['fnr'].values(), linestyle='-', marker='s', color="thistle", alpha=0.4, markersize=12, linewidth=5,
         markeredgecolor='k', zorder=3)

plt.legend(labels=[r"Surgi-2$\times$", "HKvasir"], loc='upper right', prop={'weight': 'bold', 'size': 18}, framealpha=1)

plt.plot(results['surgi4k']['size'].keys(), results['surgi4k']['fnr'].values(), linestyle='none', marker='s', color="darkviolet", alpha=1, markersize=12, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)
plt.plot(results['hkvasir']['size'].keys(), results['hkvasir']['fnr'].values(), linestyle='none', marker='s', color="thistle", alpha=1, markersize=12, linewidth=5,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)

plt.plot([22, 22], [-5, 102], linestyle='--', marker='none', color="red", linewidth=3, alpha=0.4, zorder=2)

plt.xlabel(r"Target $\alpha_{[1]}$ (PSNR)", fontsize=24, weight="bold")
plt.ylabel(r'FNR (%)', fontsize=24, weight="bold")
plt.tick_params(axis='both', which='major', labelsize=20)
ax = plt.gca()
plt.yticks([0, 20, 40, 60, 80, 100],[0, 20, 40, 60, 80, 100])
plt.xticks([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
           ["12", "", "14", "", "16", "", "18", "", "20", "", "22", "", "24", "", "26"])
ax = plt.gca().axis()
plt.gca().axis((11.5, 26.5, -5, 105))
ax = plt.gca()
for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
plt.tight_layout()
plt.grid(axis="both", linestyle='--', linewidth=2, zorder=-5)

if not os.path.exists(PATH_PLOTS +  "conf_procedure/"):
        os.makedirs(PATH_PLOTS +  "conf_procedure/")
path = PATH_PLOTS +  "conf_procedure/fnr.png"
fig.set_size_inches(6.5, 3.5)
plt.savefig(path, bbox_inches='tight')
plt.close()

# FNR vs. Mask Size in our method and theirs

fig, ax = plt.subplots(figsize=(4, 4))

plt.plot(results_ours['hkvasir']['fnr'], results_ours['hkvasir']['size'], linestyle='-', marker='o', color="red", alpha=0.4, markersize=12, linewidth=3,
         markeredgecolor='k', zorder=3)
plt.plot(results['hkvasir']['fnr'].values(), results['hkvasir']['size'].values(), linestyle='--', marker='s', color="blue", alpha=0.4, markersize=12, linewidth=3,
         markeredgecolor='k', zorder=3)

plt.legend(labels=["CFM (ours)", "Adame et al."], loc='upper right', prop={'weight': 'bold', 'size': 17}, framealpha=1)

plt.plot(results_ours['hkvasir']['fnr'], results_ours['hkvasir']['size'], linestyle='none', marker='o', color="red", alpha=1, markersize=12, linewidth=3,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)
plt.plot(results['hkvasir']['fnr'].values(), results['hkvasir']['size'].values(), linestyle='none', marker='s', color="blue", alpha=1, markersize=12, linewidth=3,
         markeredgecolor='k', markeredgewidth=1.5, zorder=3)

plt.ylabel(r"Mask (%)", fontsize=24, weight="bold")
plt.xlabel(r'$\text{FNR}_{({\tau}^{\mathrm{fail}}=22 \ \text{dB})}$ (%)', fontsize=24, weight="bold")
plt.tick_params(axis='both', which='major', labelsize=20)
ax = plt.gca()
plt.yticks([0, 20, 40, 60, 80, 100],[0, 20, 40, 60, 80, 100])
plt.xticks([0.1, 0.5, 1, 2.5, 5, 10], ["", "", "1", "2.5", "5", "10"])
ax = plt.gca().axis()
plt.gca().axis((-0.5, 12, -5, 110))
ax = plt.gca()
for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
plt.tight_layout()
plt.grid(axis="both", linestyle='--', linewidth=2, zorder=-5)

mytext = r"$\alpha_{[1]}\geq$18dB"
plt.text(0.55, 80, mytext, fontdict={'color': 'darkblue', 'weight': 'bold', 'size': 16})

mytext = r"$\alpha_{[1]}=$15dB"
plt.text(1.75, 32, mytext, fontdict={'color': 'darkblue', 'weight': 'bold', 'size': 16})

if not os.path.exists(PATH_PLOTS +  "conf_procedure/"):
        os.makedirs(PATH_PLOTS +  "conf_procedure/")
path = PATH_PLOTS +  "conf_procedure/size_fnr.png"
fig.set_size_inches(6.5, 3.5)
plt.savefig(path, bbox_inches='tight')
plt.close()