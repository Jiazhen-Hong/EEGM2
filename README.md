# EEGM2: An Efficient Mamba-2-Based Self-Supervised Framework for Long-Sequence EEG Modeling

**Authors:** 
[**Jiazhen Hong**](www.linkedin.com/in/jiazhen-hong66),
[Geoffrey Mackellar](https://www.linkedin.com/in/geoffmackellar/?originalSubdomain=au), 
[Soheila Ghane](https://www.linkedin.com/in/soheila-ghane/?originalSubdomain=au)

This work follows from the project with [**Emotiv Research**](https://www.emotiv.com/pages/enterprise), a bioinformatics research company based in Australia, and [**Emotiv**](https://www.emotiv.com/), a global technology company specializing in the development and manufacturing of wearable EEG products.

**EEGM2 Paper:** <a href="https://arxiv.org/abs/2502.17873" style="text-decoration: none;">arXiv</a>, 

## Overview
**EEGM2** shows superior performance on long-sequence tasks, where conventional models struggle.

## Project Goal
EEGM2, a self-supervised framework designed to leverage Mamba-2 blocks to accurately model sequences of various lengths in EEG signals while minimizing computational complexity for resource-limited environments. 

## Key Features
- **Multi-Branch Input Embedding**:  
- **Spatiotemporal Loss**:  
- **Mamba-2 Block**:  

## Note:

1. Mamba block used in this project was downloaded by JH on **11/14/2024** from [Mamba GitHub](https://github.com/state-spaces/mamba), only numpy < 2.x is accepted.  
2. Mamba-2 block used in this project was downloaded by JH on **12/14/2024** from [Mamba2 GitHub](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py).

```plaintext
EEGMamba/
├── code/                # Main directory for self-supervised learning and downstream tasks
│   ├── env-requirement/ # Environment backups with dates
│   ├── models/          # Includes EEGM2 and its variants
│   └── utility/         # Related functions         
├── data/                # Additional data-related code 
└── README.md            # Project overview and documentation