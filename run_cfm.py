# -----------------------------------------------------------------------------------
# Code for creating Conformal Failure Masks
# -----------------------------------------------------------------------------------

import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import json

import torch
import cv2
import numpy as np
import math

import argparse
from utils.config_reader import yaml_load
from utils.visualization import vis_psnr_heatmap_from_mse

from data import build_loaders
from uq.errornet.modeling import build_model
from sr.metrics.ssim import calculate_ssim
from sr.metrics.psnr import calculate_psnr
from data.datagen.utils import tensor2img
from uq.errornet.metrics import get_auroc, get_fpr
from uq.conformal.crc import get_lhat_crc
from uq.conformal.csuperror import get_lhat_superror

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from docs.local_data.constants import PATH_EXPERIMENTS, PATH_CONFIGS_TASKS, PATH_CONFIGS_ERRORNET

def process(opt):

    # Create dataloaders.
    loaders = build_loaders(opt)

    # Retrieve data - loop over videos.
    videos_pred, videos_ref, video_names, images_names = [], [], [], []
    print("Retrieving ... [Split:{split}]".format(split='test'))
    for idx, i_data in enumerate(loaders['test']):
        video_name = i_data['folder'][0][0]
        print("  Loading video: {video}".format(video=video_name))
        
        # Retrive data and store into dicts.
        if opt['use_source_resolution']:
            with torch.no_grad():
                videos_pred.append(torch.stack([torch.nn.functional.interpolate(i_data['prederror'][:,i,:,:].unsqueeze(1), scale_factor=1/(opt['errornet']["upscale"]), mode='area').squeeze(1) for i in range(i_data['prederror'].shape[1])], dim=1).to(torch.float16))
                videos_ref.append(torch.stack([torch.nn.functional.interpolate(i_data['gt'][:,i,:,:].unsqueeze(1), scale_factor=1/(opt['errornet']["upscale"]), mode='area').squeeze(1) for i in range(i_data['prederror'].shape[1])], dim=1).to(torch.float16))
        else:
            videos_pred.append(i_data['prederror'].to(torch.float16))
            videos_ref.append(i_data['gt'].to(torch.float16))

        video_names.append(video_name)
        images_names.append(i_data['img_names'])

        # Remove video information to save memory.
        del i_data

    # Conformal risk control experiments.
    alpha = opt['task']['alpha']
    target_psnr = opt['task']['target_psnr']

    if opt['task']['only_visualize']:
        seeds = [42]
    else:
        seeds = list(np.arange(100))

    fnr_seeds, mask_size_avg_seeds, mask_size_std_seeds, psnr_avg_seeds, psnr_std_seeds, psnr_masked_avg_seeds, psnr_masked_std_seeds = [], [], [], [], [], [], []
    for i_seed in seeds:
        torch.cuda.empty_cache()

        print("Seed ... [seed:{split}]".format(split=str(i_seed)))
        np.random.seed(i_seed)

        print("  Preparing sub-splits: [cal/test]")
        # Perform random split 70% - 30%.
        ratio_cal = 0.7
        idx = np.arange(len(videos_pred))

        # Seed split.
        idx_seed = np.copy(idx)
        np.random.shuffle(idx_seed)
        idx_calib = idx_seed[:math.ceil(len(videos_pred)*ratio_cal)]
        idx_test = idx_seed[math.ceil(len(videos_pred)*ratio_cal):]

        print("  Searching Conformal Risk Control Threshold for FNR")

        # Retrieve gt to select positive samples.
        refs_calib = torch.cat([videos_ref[i].flatten() for i in idx_calib], dim=0)
        
        # Set target PSNR and binarize GT. Positive == failure.
        with torch.no_grad():
            idx_pos_samples = torch.argwhere(((10. * torch.log10(1. / (refs_calib + 1e-8))) < target_psnr).to(torch.int16)).numpy()
        del refs_calib # Delete variables to save memory.
        
        # Search conformal prediction threshold
        if opt['conformal_procedure'] == 'crc':
    
            # Load predicted errors to set the conformalized threshold.
            preds_calib = torch.cat([videos_pred[i].flatten() for i in idx_calib], dim=0)

            # Search threshold trough conformal risk control.
            lambda_hat = get_lhat_crc(preds_calib[idx_pos_samples].to(torch.float32).numpy(), alpha, B=1)
            del preds_calib, idx_pos_samples

        elif opt['conformal_procedure'] == 'sup_error':
            lambda_hat = get_lhat_superror([videos_pred[i][:,j,:,:].flatten() for i in idx_calib for j in range(videos_pred[i].shape[1])],
                                           [videos_ref[i][:,j,:,:].flatten() for i in idx_calib for j in range(videos_ref[i].shape[1])],
                                           alpha_psnr=target_psnr)
                                    
        # Evaluate on test data.
        print("  Evaluation ... [Split:{split}]".format(split='sub-test'))

        # Retrieve preds and gt.
        preds_test = torch.cat([videos_pred[i].flatten() for i in idx_test], dim=0)
        refs_test = torch.cat([videos_ref[i].flatten() for i in idx_test], dim=0)

        with torch.no_grad():
            labels = ((10. * torch.log10(1. / (refs_test + 1e-8))) < target_psnr).to(torch.int16).numpy()
            idx_pos_samples_test = np.argwhere(labels)
        del refs_test # Delete variables to save memory.

        if opt['task']['only_visualize']:

            # Compute failure prediction scores metrics (AUC, FPR95).
            print("Computing failure prediction metrics.")
            auroc_score = get_auroc(preds_test.numpy()[labels==1], preds_test.numpy()[labels==0]) * 100
            fpr95_score = get_fpr(preds_test.numpy()[labels==1], preds_test.numpy()[labels==0]) * 100

            # Visualize histogram
            print("Creating histogram visualization of positive and negative scores.")
            from utils.visualization import plot_score_hist_compare
            plot_score_hist_compare(preds_test.numpy()[labels==1], preds_test.numpy()[labels==0],
            path=os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Errornet-' + opt['errornet']['subconfig'], opt['task']['name'] + '_histogram.png'), auc=auroc_score, fpr95=fpr95_score, title=opt['dataset']['title'])

        # FNR 
        with torch.no_grad():
            fnr_test = ((1-(preds_test[idx_pos_samples_test] >= lambda_hat).to(torch.float32))).mean().item() * 100
        del preds_test, idx_pos_samples_test, labels # Delete variables to save memory.

        # Average mask size
        mask_size, psnr, psnr_masked = [], [], []
        for i_video in range(len(idx_test)):

            # Create results folder and subfolders
            if opt['task']['only_visualize']:
                # Folder for reconstructed images ordered by videos.
                if not os.path.exists(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Errornet-' + opt['errornet']['subconfig'],'CRCMask', video_names[idx_test[i_video]])):
                    os.makedirs(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Errornet-' + opt['errornet']['subconfig'], 'CRCMask', video_names[idx_test[i_video]]))

            for i_frame in range(videos_pred[idx_test[i_video]].shape[1]):
                x_i = videos_pred[idx_test[i_video]][:,i_frame,:,:].squeeze()
                y_i = videos_ref[idx_test[i_video]][:,i_frame,:,:].squeeze()

                with torch.no_grad():
                    
                    if opt['task']['only_visualize']:
                        if opt['use_source_resolution']:
                            x_i = torch.nn.functional.interpolate(x_i.unsqueeze(0).unsqueeze(0), scale_factor=opt['errornet']["upscale"], mode='bicubic').squeeze()
                            y_i = torch.nn.functional.interpolate(y_i.unsqueeze(0).unsqueeze(0), scale_factor=opt['errornet']["upscale"], mode='bicubic').squeeze()

                    # Create uncertainty mask.
                    mask_failure = (x_i >= lambda_hat)
                    mask_clean = (x_i < lambda_hat)

                    # Compute mask size
                    mask_size.append((mask_failure.to(torch.float32).mean() * 100).item())

                    # Compute PSNR and masked PSNR
                    psnr.append((10. * torch.log10(1. / (y_i.mean() + 1e-8))).item())
                    psnr_masked.append((10. * torch.log10(1. / (y_i[mask_clean].mean() + 1e-8))).item())
                    
                    torch.cuda.empty_cache()

                # Visualization
                if opt['task']['only_visualize']:
                    mask_np = mask_failure.to(torch.float32).numpy().astype(np.float32) * 255
                    cv2.imwrite(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Errornet-' + opt['errornet']['subconfig'], 'CRCMask', video_names[idx_test[i_video]], images_names[idx_test[i_video]][i_frame][0].replace("pt",'png')), mask_np.astype(np.uint8))

        print("  Split performance: FNR={fnr:.2f}% -- PSNR/PSNR-M={psnr:.2f}/{psnrmask:.2f} -- Mask Size={masksize:.2f}%".format(
            fnr=fnr_test, psnr=np.mean(psnr), psnrmask=np.mean(psnr_masked), masksize=np.mean(mask_size)))

        # Store tracked metrics accross seeds.
        fnr_seeds.append(fnr_test)
        mask_size_avg_seeds.append(np.mean(mask_size))
        mask_size_std_seeds.append(np.std(mask_size))
        psnr_avg_seeds.append(np.mean(psnr))
        psnr_std_seeds.append(np.std(psnr))
        psnr_masked_avg_seeds.append(np.mean(psnr_masked))
        psnr_masked_std_seeds .append(np.std(psnr_masked))

    print("Average across experiments: FNR={fnr:.2f}% -- PSNR={psnr_avg:.2f}+-{psnr_std:.2f} -- PSNR-M={psnr_m_avg:.2f}+-{psnr_m_std:.2f} -- Mask Size={masksize_avg:.2f}+-{masksize_std:.2f}%".format(
    fnr=np.median(fnr_seeds), psnr_avg=np.median(psnr_avg_seeds), psnr_std=np.median(psnr_std_seeds), psnr_m_avg=np.median(psnr_masked_avg_seeds), psnr_m_std=np.median(psnr_masked_std_seeds), masksize_avg=np.median(mask_size_avg_seeds), masksize_std=np.median(mask_size_std_seeds)))

    # Save split results in dict
    metrics = {
        'mean': 
            {'fnr_seeds': np.round(np.mean(fnr_seeds), 2), 
            'psnr_avg_seeds': np.round(np.mean(psnr_avg_seeds), 1),
            'psnr_std_seeds': np.round(np.mean(psnr_std_seeds), 1),
            'psnr_masked_avg_seeds': np.round(np.mean(psnr_masked_avg_seeds), 1),
            'psnr_masked_std_seeds': np.round(np.mean(psnr_masked_std_seeds), 1),
            'mask_size_avg_seeds': np.round(np.mean(mask_size_avg_seeds), 1),
            'mask_size_std_seeds': np.round(np.mean(mask_size_std_seeds), 1)},
        'median': 
            {'fnr_seeds': np.round(np.median(fnr_seeds), 2), 
            'psnr_avg_seeds': np.round(np.median(psnr_avg_seeds), 1),
            'psnr_std_seeds': np.round(np.median(psnr_std_seeds), 1),
            'psnr_masked_avg_seeds': np.round(np.median(psnr_masked_avg_seeds), 1),
            'psnr_masked_std_seeds': np.round(np.median(psnr_masked_std_seeds), 1),
            'mask_size_avg_seeds': np.round(np.median(mask_size_avg_seeds), 1),
            'mask_size_std_seeds': np.round(np.median(mask_size_std_seeds), 1)},
    }
    with open(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Errornet-' + opt['errornet']['subconfig'], 'results_crc' + '_psnr_' + str(int(opt['task']['target_psnr'])) +  '_alpha_' + str((opt['task']['alpha'])).replace(".","") + '.txt'), "w") as file: 
        json.dump(metrics, file, indent=1, sort_keys=False)
    
    return

def main():

    # Parse task config file.
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='surgisr4k_highres_swinir.yml',
                        help='surgisr4k_lowres_swinir - surgisr4k_highres_swinir - hyperkvasir_basicvsr')
    parser.add_argument('--errornet_config', default='errornet_2layer.yml')
    parser.add_argument('--conformal_procedure', default='crc', help='crc - sup_error')
    parser.add_argument('--only_visualize', default=False, type=bool)
    parser.add_argument('--alpha', default=0.10, type=float)
    parser.add_argument('--target_psnr', default=22.0, type=float)
    parser.add_argument('--use_source_resolution', default=True, type=bool)
    args, unknown = parser.parse_known_args()

    # Parse task yml to dict.
    opt = yaml_load(PATH_CONFIGS_TASKS + args.task)
    opt['task']['only_visualize'] = args.only_visualize
    opt['task']['alpha'] = args.alpha
    opt['task']['target_psnr'] = args.target_psnr
    opt['use_source_resolution'] = args.use_source_resolution
    opt['conformal_procedure'] = args.conformal_procedure

    # Parse dataset and model configs to the options.
    opt['dataset'] = yaml_load(opt['task']['dataset_config'])['dataset']
    opt['network'] = yaml_load(opt['task']['sr_model_config'])['network']
    opt['errornet'] = yaml_load(PATH_CONFIGS_ERRORNET + args.errornet_config)['network']
    opt['errornet_train'] = yaml_load(PATH_CONFIGS_ERRORNET + args.errornet_config)['training']
    opt['errornet']["upscale"] = opt['dataset']['scale_factor']
    opt['errornet']["embed_dim"] = opt['network']['embed_dim']

    # Add type of task to address
    opt['task']['type'] = 'CRC'

    # Run code.
    process(opt=opt)

if __name__ == "__main__":
    main()
