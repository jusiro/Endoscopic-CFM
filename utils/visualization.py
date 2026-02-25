import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
plt.rcParams['font.weight'] = 'bold'
import seaborn as sns
import numpy as np
# configs
sns.set_theme(style="ticks")

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def vis_psnr_heatmap_from_mse(mse_map, vis_path, max_db=48, smooth=True):
    heatmap = mse_map

    # Produce PSNR map and corret ill-defined numbers
    heatmap = 10. * torch.log10(1. / (torch.nn.functional.relu(mse_map) + 1e-8))
    heatmap[heatmap>max_db]=max_db
    heatmap = heatmap.numpy()

    # Smooth mse with local window (for visualization porpuses)
    if smooth:
        w = 5
        kernel = np.ones((1,1,w,w))/(w*w)
        with torch.no_grad():
            heatmap = torch.nn.functional.conv2d(input=torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device),
                                                weight=torch.tensor(kernel).to(torch.float32).to(device), stride=1, padding='same').squeeze().cpu()
            torch.cuda.empty_cache()    

    # Visualize heatmap
    plt.imshow(heatmap, cmap='autumn', interpolation='nearest', vmin=0, vmax=max_db)
    im_ratio = heatmap.shape[0]/heatmap.shape[1]
    plt.colorbar(fraction=0.047*im_ratio)
    plt.axis("off")
    plt.savefig(vis_path, bbox_inches='tight')
    plt.close()


def plot_score_hist_compare(pos_scores, neg_scores, path, bins=40, kde=True, title=None, auc=None, fpr95=None):
    pos_scores = np.asarray(pos_scores, dtype=float)
    neg_scores = np.asarray(neg_scores, dtype=float)
    pos_scores[pos_scores<0] = 0.
    neg_scores[neg_scores<0] = 0.
    neg_scores = np.random.choice(neg_scores, size=len(pos_scores))
    # Compute percentile x for limits
    xlim = np.quantile(pos_scores, 0.95)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    sns.histplot(x=pos_scores, color="red", label="PSNR<22", kde=kde, stat="probability",
                 line_kws={"linewidth": 3}, bins=bins, common_norm=False, binrange=(0, xlim))
    sns.histplot(x=neg_scores, color="green", label="PSNR>22", kde=kde, stat="probability",
                 line_kws={"linewidth": 3}, bins=bins, common_norm=False, binrange=(0, xlim))

    # Legends and axis
    if title is not None:
        plt.title(title, fontsize=25, weight="bold")
    plt.legend(loc='upper right', prop={'weight': 'bold', 'size': 18}, framealpha=1)
    plt.xlabel(r"Failure score", fontsize=25, weight="bold")
    plt.ylabel('Density', fontsize=25, weight="bold")
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tick_params(axis='both', which='major', labelsize=23)
    axis = plt.gca().axis()
    ax = plt.gca()
    plt.grid(axis="both", linestyle='--', linewidth=3, zorder=-1)
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')

    ax.set_ylim([0, 0.6])
    x_axi_lim = ax.get_xlim()
    ax.set_xlim([-0.00001, xlim])

    plt.xticks(np.arange(0, xlim+xlim/4, step=xlim/4).tolist(),labels=[])

    # Metrics box
    txt = (
        f"AUROC = {auc:.2f}\n" + f"FPR95 = {fpr95:.2f}"
    )
    plt.text(0.5, 0.45, txt, ha='left', va='top', transform=ax.transAxes,
     fontdict={'weight': 'bold', 'size': 18}, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.8', alpha=0.9))

    plt.tight_layout()
    fig.set_size_inches(6.5, 3.5)

    plt.savefig(path, bbox_inches='tight')
    plt.close()