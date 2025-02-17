# EEGMamba

## Overview
**EEGMamba** is a research project focused on building a foundation model for Electroencephalography (EEG) signal processing using Mamba. 
## Project Goal
The main objective of EEGMamba is to implement and optimize the **Mamba** and Transformer-based architecture for EEG signal processing. The project aims to create a lightweight, scalable, and efficient foundation model to handle downstream tasks of EEG.

## Key Features
- **Lightweight Model**: Exploring new approaches such as Mamba and S4 for building efficient models.
- **Achieved**: 
- **Edge Deployment**: Building models that can be deployed on edge devices for real-time applications.

## Current Status
- Initial setup and repository creation.
- Literature review on existing models.
- Test U-Net, Single Mamba Block (have gradient explosion problem)
- Built "Mentality", fix the gradient explosion problem.
- We have built "Mentality" using a custom Mamba block with a normalization layer and ResNet, similar to the “Mamba4Rec” architecture.

## Note:

1. Mambm block used in this project is downloading on 11/14/2024 form #https://github.com/state-spaces/mamba, only numpy < 2.x is apcceted.
2. 


```plaintext
EEGMamba/
├── code/                # Main directory for self-supervised learning and downstream tasks
│   ├── env-requirement/ # Environment backups with dates
│   ├── models/          # Includes Mentality and other models (future)
│   ├── utility/         # Filters or other related functions
│   └── data/            # Additional data-related code (if applicable)
├── data/                # Sample EEG datasets (to be added)
├── src/                 # Source code for model development
├── notebooks/           # Jupyter notebooks for experiments and visualizations during the test phase
├── results/             # Directory for storing results, TensorBoard logs, dates, and outputs from self-supervised learning and classifiers
└── README.md            # Project overview and documentation



