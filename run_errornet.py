# -----------------------------------------------------------------------------------
# Code for training errornet and making predictions on test data
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
from uq.errornet.modeling import build_model
from sr.metrics.ssim import calculate_ssim
from sr.metrics.psnr import calculate_psnr
from data.datagen.utils import tensor2img

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from docs.local_data.constants import PATH_EXPERIMENTS, PATH_CONFIGS_TASKS, PATH_CONFIGS_ERRORNET


def process(opt):

    # Create dataloaders.
    loaders = build_loaders(opt)

    # Create Errornet model.
    model = build_model(opt).to(device).float()
    model.train()

    if not opt['task']['only_test']:
        
        # Training
        epochs = opt['errornet_train']['epochs']
        lr = float(opt['errornet_train']['lr'])

        # Set training optimizer
        optim = torch.optim.Adam(params=model.parameters(), lr=lr)
        # Set scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=epochs)

        for i_epoch in range(epochs):
            # Loop over videos.
            for idx, i_data in enumerate(loaders['train']):
                tracking_loss = 0.0

                # Loop over frames.
                for i_frame in range(i_data['feats'].shape[1]):

                    # Retrieve frame.
                    x_i = i_data['feats'][:,i_frame,:,:,:].to(torch.float32).to(device)
                    y_i = i_data['gt'][:,i_frame,:,:].to(torch.float32).to(device)

                    # Forward.
                    pred_i = model(x_i)

                    # Compute loss: mse
                    loss = torch.mean((pred_i.squeeze() - y_i.squeeze()).pow(2))

                    # Update model.
                    loss.backward()
                    optim.step()
                    optim.zero_grad()
                    torch.cuda.empty_cache()

                    # Running loses
                    tracking_loss += loss.item() / i_data['feats'].shape[1]

                # Track training
                print("Epoch {i_epoch}/{epochs} -- Video {i_video}/{videos} -- loss={loss:.8f}".format(
                    i_epoch=i_epoch + 1, epochs=epochs, i_video=idx + 1, videos=len(loaders['train']), loss=round(tracking_loss, 8)), end="\n")

            # Update scheduler at the end of epoch.
            scheduler.step() 

        if not os.path.exists(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Errornet-' + opt['errornet']['subconfig'])):
            os.makedirs(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Errornet-' + opt['errornet']['subconfig']))

        # Save weights
        torch.save(model.state_dict(), os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Errornet-' + opt['errornet']['subconfig'], 'errornet.pth'))

    if opt['task']['only_test']:
        print("Loading weights from:", os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Errornet-' + opt['errornet']['subconfig'], 'errornet.pth'))
        ckpt = torch.load(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Errornet-' + opt['errornet']['subconfig'], 'errornet.pth'))
        model.load_state_dict(ckpt)

    # Run predictions in test data and save outputs
    print("Testing ... [Split:{split}]".format(split='test'))
    model.eval()
    for idx, i_data in enumerate(loaders['test']):
        video_name = i_data['folder'][0][0]
        print("  Video: {video}".format(video=video_name))

        # Folder for predicted error map.
        if not os.path.exists(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Errornet-' + opt['errornet']['subconfig'], 'PredError', video_name)):
            os.makedirs(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Errornet-' + opt['errornet']['subconfig'], 'PredError', video_name))
        # Path with visualizations.
        if not os.path.exists(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Errornet-' + opt['errornet']['subconfig'], 'PredErrorVis', video_name)):
            os.makedirs(os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Errornet-' + opt['errornet']['subconfig'], 'PredErrorVis', video_name))

        # Loop over frames.
        for i_frame in range(i_data['feats'].shape[1]):
            # Retrieve frame.
            x_i = i_data['feats'][:,i_frame,:,:,:].to(torch.float32).to(device)
            y_i = i_data['gt'][:,i_frame,:,:].to(torch.float32).cpu().squeeze()

            # Forward.
            with torch.no_grad():
                pred_i = model(x_i).cpu().squeeze()
                loss = torch.mean((pred_i.squeeze() - y_i.squeeze()).pow(2))
                torch.cuda.empty_cache()

            # Save predicted error.
            torch.save(pred_i.to(torch.float16), os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Errornet-' + opt['errornet']['subconfig'], 'PredError', video_name, i_data['img_names'][i_frame][0].replace('.png', '.pt')))

            # Visualize outputs (in terms of predicted psnr).
            vis_psnr_heatmap_from_mse(pred_i, os.path.join(PATH_EXPERIMENTS, opt['task']['name'], opt['network']['name'], 'Errornet-' + opt['errornet']['subconfig'], 'PredErrorVis', video_name, i_data['img_names'][i_frame][0].replace('.pt', '.png')),
                                      smooth=False)

    return

def main():

    # Parse task config file.
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='surgisr4k_highres_swinir.yml',
                        help='surgisr4k_lowres_swinir - surgisr4k_highres_swinir - hyperkvasir_basicvsr')
    parser.add_argument('--errornet_config', default='errornet_2layer.yml')
    parser.add_argument('--only_test', default=False, type=bool)
    args, unknown = parser.parse_known_args()

    # Parse task yml to dict.
    opt = yaml_load(PATH_CONFIGS_TASKS + args.task)
    opt['task']['only_test'] = args.only_test

    # Parse dataset and model configs to the options.
    opt['dataset'] = yaml_load(opt['task']['dataset_config'])['dataset']
    opt['network'] = yaml_load(opt['task']['sr_model_config'])['network']
    opt['errornet'] = yaml_load(PATH_CONFIGS_ERRORNET + args.errornet_config)['network']
    opt['errornet_train'] = yaml_load(PATH_CONFIGS_ERRORNET + args.errornet_config)['training']
    opt['errornet']["upscale"] = opt['dataset']['scale_factor']
    opt['errornet']["embed_dim"] = opt['network']['embed_dim']

    # Add type of task to address
    opt['task']['type'] = 'PredictError'

    # Run code.
    process(opt=opt)


if __name__ == "__main__":
    main()