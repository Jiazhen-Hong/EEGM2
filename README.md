# EEGM2

## Overview
**EEGM2** shows superior performance on long-sequence tasks, where conventional models struggle.
## Project Goal
EEGM2, a self-supervised framework designed to leverage Mamba-2 blocks to accurately model sequences of various lengths in EEG signals while minimizing computational complexity for resource-limited environments. 
## Key Features
- **Multi-Branch Input Embedding**: 
- **Spatiotemporal Loss**: 
- **Mamba-2 Block**: 

## Note:

1. Mambm block used in this project is downloaded by JH 11/14/2024 from #https://github.com/state-spaces/mamba, only numpy < 2.x is apcceted.
2. Mambm2 block used in this project is downloaded by JH 12/14/2024, from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py


```plaintext
EEGMamba/
├── code/                # Main directory for self-supervised learning and downstream tasks
│   ├── env-requirement/ # Environment backups with dates
│   ├── models/          # Includes EEGM2 and its vairant
│   └──utility/          # Related functions         
├── data/                # Additional data-related code 
└── README.md            # Project overview and documentation



