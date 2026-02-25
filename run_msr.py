# -----------------------------------------------------------------------------------
# Code for runing medical super-resolution
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

import argparse
from utils.config_reader import yaml_load
from utils.visualization import vis_psnr_heatmap_from_mse

from data import build_loaders
from sr.modeling import build_model
from sr.metrics.ssim import calculate_ssim
from sr.metrics.psnr import calculate_psnr
from data.datagen.utils import tensor2img

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from docs.local_data.constants import PATH_EXPERIMENTS, PATH_CONFIGS_TASKS

def process(opt):

    # Create folders for the experiment.
    if not os.path.exists(PATH_EXPERIMENTS):
        os.makedirs(PATH_EXPERIMENTS)
    if not os.path.exists(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'])):
        os.makedirs(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name']))

    # Create dataloaders.
    loaders = build_loaders(opt)

    # Create SR model and init weights.
    model = build_model(opt).to(device).float()

    # Run predicitons and save outputs (reconstructed images, intermediate features).
    for split in loaders: # Loop over data splits.

        # Init metrics dict.
        print("[Split:{split}]".format(split=split))
        metrics = {}
        
        # Loop over videos.
        for idx, i_data in enumerate(loaders[split]):
            video_name = i_data['folder'][0][0]
            print("  Video: {video}".format(video=video_name))
            metrics[video_name] = {'psnr': [], 'ssim': []}

            # Create results folder and subfolders
            if opt['task']['save_outputs']:
                # Folder for reconstructed images ordered by videos.
                if not os.path.exists(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Preds', video_name)):
                    os.makedirs(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Preds', video_name))
                # Folder for intermediate features ordered by videos.
                if not os.path.exists(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Features', video_name)):
                    os.makedirs(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Features', video_name))
                # Folder for target error map.
                if not os.path.exists(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Error', video_name)):
                    os.makedirs(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Error', video_name))
                # Folder for visualization of error map in terms of PSNR.
                if not os.path.exists(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'ErrorVis', video_name)):
                    os.makedirs(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'ErrorVis', video_name))

            # Loop over frames.
            for i_frame in range(i_data['imgs_lq'].shape[1]):
                
                # Forward frame.
                with torch.no_grad():
                    if opt['network']['name'] == 'Interpolation':
                        i_frame_pred = model(i_data['imgs_lq'][:,i_frame,:,:,:].to(device))
                        i_frame_feats = None
                    else:
                        i_frame_pred, i_frame_feats = model(i_data['imgs_lq'][:,i_frame,:,:,:].to(device), return_feats=True)
                        i_frame_pred = i_frame_pred.cpu()
                        i_frame_feats = i_frame_feats.cpu()
                    torch.cuda.empty_cache()
                
                # Compute metrics
                with torch.no_grad():
                    psnr, mse_map = calculate_psnr(i_data['imgs_gt'][:,i_frame,:,:,:].to(device), i_frame_pred.to(device))
                    ssim, ssim_map = calculate_ssim(i_data['imgs_gt'][:,i_frame,:,:,:].to(device), i_frame_pred.to(device))
                    torch.cuda.empty_cache()

                # Save results (predicted image, features, error map, etc.)
                if opt['task']['save_outputs']:
                    # Reconstructed images.
                    cv2.imwrite(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Preds', video_name, i_data['img_names'][i_frame][0]), tensor2img(i_frame_pred))
                    # Intermediate features.
                    if i_frame_feats != None:
                        torch.save(i_frame_feats.to(torch.float16), os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Features', video_name, i_data['img_names'][i_frame][0].replace('.png', '.pt')))
                    # Error map.
                    torch.save(mse_map.to(torch.float16), os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Error', video_name, i_data['img_names'][i_frame][0].replace('.png', '.pt')))
                    # Visualization of error map
                    vis_psnr_heatmap_from_mse(mse_map, os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'ErrorVis', video_name, i_data['img_names'][i_frame][0]))

                # Track metrics per frame.
                metrics[video_name]['psnr'].append(np.round(psnr, 2).item())
                metrics[video_name]['ssim'].append(np.round(ssim*100, 2).item())
                print("     Frame: {frame} - PSNR/SSIM={psnr:.2f}/{ssim:.2f}".format(frame=i_data['img_names'][i_frame][0], psnr=psnr, ssim=ssim*100))

            # Track metrics per video.
            metrics[video_name]['psnr_avg'] = np.round(np.mean(metrics[video_name]['psnr']), 2).item()
            metrics[video_name]['ssim_avg'] = np.round(np.mean(metrics[video_name]['ssim']), 2).item()
            print("     Average video: PSNR/SSIM={psnr:.2f}/{ssim:.2f}".format(psnr=np.mean(metrics[video_name]['psnr']), ssim=np.mean(metrics[video_name]['ssim'])))
        
        # Track average metric per partition - average per video.
        psnr_split_per_video = np.round(np.mean([metrics[i_video]['psnr_avg'] for i_video in metrics.keys()]), 2).item()
        ssim_split_per_video = np.round(np.mean([metrics[i_video]['ssim_avg'] for i_video in metrics.keys()]), 2).item()
        psnr_split_per_video_std = np.round(np.std([metrics[i_video]['psnr_avg'] for i_video in metrics.keys()]), 2).item()
        ssim_split_per_video_std = np.round(np.std([metrics[i_video]['ssim_avg'] for i_video in metrics.keys()]), 2).item()

        # Track average metric per partition - average per image.
        psnr_split_per_image, ssim_split_per_image= [], []
        for i_video in metrics.keys():
            for i_frame in range(len(metrics[i_video]['psnr'])):
                psnr_split_per_image.append(metrics[i_video]['psnr'][i_frame])
                ssim_split_per_image.append(metrics[i_video]['ssim'][i_frame])
        psnr_split_per_image_avg = np.round(np.mean(psnr_split_per_image), 2).item()
        ssim_split_per_image_avg = np.round(np.mean(ssim_split_per_image), 2).item()
        psnr_split_per_image_std = np.round(np.std(psnr_split_per_image), 2).item()
        ssim_split_per_image_std = np.round(np.std(ssim_split_per_image), 2).item()

        # Store in output dict.
        metrics['psnr_split_video'] = psnr_split_per_video
        metrics['psnr_split_per_video_std'] = psnr_split_per_video_std
        metrics['ssim_split_video'] = ssim_split_per_video
        metrics['ssim_split_per_video_std'] = ssim_split_per_video_std
        metrics['psnr_split_per_image'] = psnr_split_per_image_avg
        metrics['psnr_split_per_image_std'] = psnr_split_per_image_std
        metrics['ssim_split_per_image'] = ssim_split_per_image_avg
        metrics['ssim_split_per_image_std'] = ssim_split_per_image_std

        # Save split results in dict
        with open(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'results_' + split + '.txt'), "w") as file: 
            json.dump(metrics, file, indent=1, sort_keys=False)

    return

def main():

    # Parse task config file.
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='surgisr4k_highres_swinir.yml',
                        help='surgisr4k_lowres_swinir - surgisr4k_highres_swinir - hyperkvasir_basicvsr')
    parser.add_argument('--save_outputs', default=True, type=bool)
    args, unknown = parser.parse_known_args()

    # Parse task yml to dict.
    opt = yaml_load(PATH_CONFIGS_TASKS + args.task)
    opt['task']['save_outputs'] = args.save_outputs

    # Parse dataset and model configs to the options.
    opt['dataset'] = yaml_load(opt['task']['dataset_config'])['dataset']
    opt['network'] = yaml_load(opt['task']['sr_model_config'])['network']

    # Add type of task to address
    opt['task']['type'] = 'SR'

    # Run code.
    process(opt=opt)


if __name__ == "__main__":
    main()