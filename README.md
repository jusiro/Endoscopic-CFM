# Trustworthy Endoscopic Super-Resolution

### Install

* Install in your environment a compatible torch version with your GPU. For example:

```
conda create -n cfm python=3.11 -y
conda activate cfm
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

```
git clone https://github.com/******/Endoscopic-CFM.git
cd Endoscopic-CFM
pip install -r requirements.txt
```

### Preparing the datasets
- Configure data paths (see [`./docs/local_data/constants.py`](./docs/local_data/constants.py)).
- Download, and configure datasets (see [`./docs/local_data/datasets/README.md`](./docs/local_data/datasets/README.md)).

### Preparing SR models
- Download, and configure SR models (see [`.../models_weights/README.md`](./docs/local_data/models_weights/README.md)).

## Usage
We present the basic usage here.

(a) Run predictions using the super-resolution model:
- `python run_msr.py --task surgisr4k_lowres_swinir.yml`

(b) Train the Reconstruction Error Network:
- `python run_errornet.py --task surgisr4k_lowres_swinir.yml --errornet_config errornet_2layer.yml`

(c) Create and validate the Conformal Failure Masks:
- `python run_cfm.py --task surgisr4k_lowres_swinir.yml --errornet_config errornet_2layer.yml --target_psnr 22 --alpha 0.05`

You will find the final and intermediate results at [`./docs/local_data/experiments/`](./docs/local_data/experiments/).

## Citation

If you find this repository useful, please consider citing the following sources.

```
@inproceedings{sstextu,
    title={Trustworthy Endoscopic Super-Resolution},
    author={******},
    booktitle={arXiv preprint arXiv:xxxx.xxxxx},
    year={2026}
}
```
