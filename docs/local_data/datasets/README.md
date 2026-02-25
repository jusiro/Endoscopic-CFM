We suggest the following dataset organization to ease management and avoid modifying the source code. The datasets structure looks like:

```
Endoscopic-CFM/
└── docs/local_data/
         └── datasets
             ├── Hyperkvasir/
             └── SurgiSR4K/
```

In the following, we provide specific download links and expected structure for each individual dataset.

### HyperKvasir - [LINK](https://huggingface.co/datasets/jeffrey423/MedVSR_dataset/)

```
.
└── Hyperkvasir/
    ├── hyperkvasir_test/
    │   ├── BIx4/
    │   │   ├── 0251/
    │   │   │   ├── 0000.png
    │   │   │   └── ...
    │   │   └── ...
    │   └── GT/
    │       ├── 0251/
    │       │   ├── 0000.png
    │       │   └── ...
    │       └── ...
    ├── hyperkvasir_training/
    │   └── ...
    ├── hyperkvasir_val/
    │   └── ...
    └── meta_info_hyperkvasir.txt
```

### SurgiSR4K - [LINK](https://www.synapse.org/Synapse:syn68756003)

```
.
└── SurgiSR4K/
    ├── data/
    │   ├── images/
    │   │   ├── 480x270p/
    │   │   │   ├── vid_001_480x270p_1tool/
    │   │   │   │   ├── vid_001_480x270p_1tool_1.png
    │   │   │   │   └── ...
    │   │   │   └── ...
    │   │   ├── 960x540p/
    │   │   │   └── ...
    │   │   └── 3840x2160p/
    │   │       └── ...
    │   └── videos/
    │       └── ...
    ├── scripts/
    │   └── ...
    ├── DATASET_ORGANIZATION.md
    ├── LICENSE
    └── README.md
```