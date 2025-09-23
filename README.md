# An Efficient Self-Supervised Framework for Long-Sequence EEG Modeling

- **2025-06-06**: Repository anonymized to comply with the triple-blind reviewing policy of the ICDM conference.  
- **2025-09-21**: Paper accepted at the **AI4TS Workshop @ ICDM 2025**.  
- Paper is in proceedings. Preprint available on [arXiv](https://arxiv.org/abs/2502.17873).

## Overview

EEGM2 consists of two stages:  
1. **Self-supervised pretraining**  
2. **Downstream classification tasks**

Pretrained models are provided in the `Pretrained_EEGM2/` folder.

Dataset structure and preparation guidelines are described in the `Datasets/` folder.

Model parameters and training settings can be configured in `config.json` (for pretraining) and `config_downstream.json` (for downstream tasks).

## Setup

Instructions refer to Unix-based systems (e.g., Linux, macOS).

This code has been tested with **Python 3.10.16**. For users who prefer using Conda, environment YAML files are provided.

Create an environment from a specific file:
```bash
conda env create -f Env-requirement/Mamba2-env20250206.yml
```

## Run Commands

### 1. Pretrain EEGM2 on TUAB (e.g., 30-second segments)
```bash
python SelfSupervised.py --config config.json
```
### 2. Run downstream classification on Crowdsourced dataset
```bash
python Downstream.py --config config_downstream.json
```

## Results Summary (TUAB Dataset)
Balanced Accuracy and AUROC across varying input sequence lengths.

| Duration       | Model           | Balanced ACC      | AUROC         |
|----------------|------------------|--------------------|---------------|
| **10 Seconds** | EEGM2 (Light)    | 79.14 ± 0.21       | 0.8559 ± 0.000 |
|                | EEGM2 (Fine)     | **80.87 ± 0.54**   | **0.8864 ± 0.000** |
| **30 Seconds** | EEGM2 (Light)    | 78.97 ± 0.25       | 0.8575 ± 0.000 |
|                | EEGM2 (Fine)     | **81.71 ± 0.12**   | **0.8932 ± 0.000** |
| **60 Seconds** | EEGM2 (Light)    | 76.94 ± 0.33       | 0.8257 ± 0.000 |
|                | EEGM2 (Fine)     | **80.68 ± 0.45**   | **0.8803 ± 0.000** |
| **100 Seconds**| EEGM2 (Light)    | 74.57 ± 0.27       | 0.7986 ± 0.000 |
|                | EEGM2 (Fine)     | **81.08 ± 0.28**   | **0.8869 ± 0.000** |
