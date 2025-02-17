"""
Created on Mon Nov 15 10:12:39 2024

@author: jiazhen@emotiv.com
"""

import numpy as np
import os, json
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from SelfSupervised_Trainer import *
from models import EEGM2, EEGM2_S1, EEGM2_S3, EEGM2_S4, EEGM2_S5, UNet

from utility import chebyBandpassFilter, setup_logging, create_data_loaders, TUABLoader, MSEWithSpectralLoss, L1WithSpectralLoss


# -------------------------------
# Configuration and Logging Setup
# -------------------------------
data_name = 'TUAB' #Alpha, Attention, Crowdsource, STEW, DriverDistraction, DREAMER, TUAB
model_name = "EEGM2"  

TU_sequen = 12800  # 256, 1280, 3840, 7680, 12800

# Load json for hyper-parameter
with open("config.json", "r") as f:
    config = json.load(f)
# Define result path & log path
result_path =  f'../results_paper/{data_name}/{datetime.now().strftime("%Y%m%d")}/{model_name}-{datetime.now().strftime("%H%M%S")}/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
logger, log_file = setup_logging(result_path)
logger.info(f"Loaded hyperparameters: [{config}]. \n")
logger.info(f"{data_name} \n")

if "cuda:0" in config["GPU_device"]:
    torch.cuda.set_per_process_memory_fraction(0.9, device=0)
elif "cuda:1" in config["GPU_device"]:
    torch.cuda.set_per_process_memory_fraction(0.9, device=1)
device = torch.device(config["GPU_device"])
logger.info(f"GPU: {device}\n model name: {model_name}")

###########################
# --   Data Loading   -- 
###########################

if data_name in ['Alpha', 'Attention', 'Crowdsource', 'STEW', 'DriverDistraction', 'DREAMER']:
    if data_name == 'DREAMER':
        data_x_dir = np.load(f'/data/data_downstream_task/{data_name}/{data_name}.npy', allow_pickle=True)
    else:
        data_x_dir = np.load(f'/data/data_downstream_task/{data_name}/{data_name}.npy', allow_pickle=True).item()
    logger.info("="*80)
    logger.info(f"Successful Load the Dataset: {data_name}")
    logger.info("="*80)
    train_data, train_label, test_data, test_label, val_data, val_label, All_train_data, All_train_label = \
    data_x_dir['train_data'], data_x_dir['train_label'], data_x_dir['test_data'], data_x_dir['test_label'], \
    data_x_dir['val_data'], data_x_dir['val_label'], data_x_dir['All_train_data'], data_x_dir['All_train_label']

    total_trials = All_train_data.shape[0] + test_data.shape[0]
    train_percentage = (train_data.shape[0] / total_trials) * 100
    val_percentage = (val_data.shape[0] / total_trials) * 100
    test_percentage = (test_data.shape[0] / total_trials) * 100
    logger.info(f"Total number of trails: {total_trials}")
    logger.info(f"  - Train data shape: {train_data.shape}, Train label shape: {train_label.shape} ({train_percentage:.2f}%)")
    logger.info(f"  - Validation data shape: {val_data.shape}, Validation label shape: {val_label.shape} ({val_percentage:.2f}%)")
    logger.info(f"  - Test data shape: {test_data.shape}, Test label shape: {test_label.shape} ({test_percentage:.2f}%)")
    logger.info(f"  - All train data shape: {All_train_data.shape}, All train label shape: {All_train_label.shape}\n{'-'*80}")
    # Applying chebyBandpassFilter
    if config["filter"]:
        train_data = chebyBandpassFilter(train_data, [0.2, 0.5, 40, 48], gstop=40, gpass=1, fs=128)
        logger.info('training set filtered.')
        val_data = chebyBandpassFilter(val_data, [0.2, 0.5, 40, 48], gstop=40, gpass=1, fs=128)
        logger.info('validation set filtered.')
        test_data = chebyBandpassFilter(test_data, [0.2, 0.5, 40, 48], gstop=40, gpass=1, fs=128)
        logger.info('testing set filtered.')
        logger.info("-"*80)

    # dataset loader and info print.
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data=train_data,
        train_label=train_label,
        val_data=val_data,
        val_label=val_label,
        test_data=test_data,
        test_label=test_label,
        batch_size=config["batch_size"]
    )
elif data_name in ['TUAB']:
    tuab_root = f"/data/datasets_public/TUAB/edf/processed_128hz_{TU_sequen}seqlen_JH/"
    train_files = os.listdir(os.path.join(tuab_root, "train"))
    val_files = os.listdir(os.path.join(tuab_root, "val"))
    test_files = os.listdir(os.path.join(tuab_root, "test"))
    # Create dataset objects
    train_data = TUABLoader((os.path.join(tuab_root, "train"), train_files))
    val_data = TUABLoader((os.path.join(tuab_root, "val"), val_files))
    test_data = TUABLoader((os.path.join(tuab_root, "test"), test_files))
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, num_workers=32, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config["batch_size"], shuffle=False, num_workers=32, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config["batch_size"], shuffle=False, num_workers=32, persistent_workers=True)
    logger.info("Loaded TUAB dataset.")
else:
    raise ValueError(f"Unknown data_name: {data_name}")
# Print information about datasets and batches
for name, loader in [("Training", train_loader), ("Validation", val_loader), ("Testing", test_loader)]:
    num_batches, total_trails = len(loader), len(loader.dataset)
    logger.info(f"{name} set: Number of batches: {num_batches}")
    logger.info(f"{name} set: Total number of trails: {total_trails}\n")
data_batch = next(iter(train_loader))
logger.info(f"Shape for each batch: {data_batch[0].shape}")
in_times, in_channels = data_batch[0].shape[2], data_batch[0].shape[1]
logger.info(f"    # Duration (s): {in_times}   # of channels: {in_channels}\n")

#######################################
#      Training: Reconstruction Task
#######################################

if model_name == "EEGM2":
    model = EEGM2(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"]).to(device)
elif model_name == "EEGM2_S1":
    model = EEGM2_S1(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"]).to(device)
elif model_name == "EEGM2_S3":
    model = EEGM2_S3(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"]).to(device)
elif model_name == "EEGM2_S4":
    model = EEGM2_S4(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"]).to(device)
elif model_name == "EEGM2_S5":
    model = EEGM2_S5(in_channels, in_channels, d_state=config["d_state"], d_conv=config["d_conv"], expand=config["expand"]).to(device)
elif model_name == "UNet":
    model = UNet(in_channels, in_channels).to(device)

else:
    raise ValueError("model name does not exist.")

if config['loss'] == "L1Loss":  
    criterion = nn.L1Loss() #MSELoss(),SmoothL1Loss()
elif config['loss'] == "L1WithSpectralLoss":
    criterion = L1WithSpectralLoss()
elif config['loss'] == "MSEWithSpectralLoss":
    criterion = MSEWithSpectralLoss()
else:
    raise ValueError("loss name does not exist.")

optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
trainer = Trainer(model, criterion, optimizer, num_epochs=config["num_epochs"], log_dir=result_path)
# model information:
for name, param in model.named_parameters():
    if "input_embedding" in name: #or "mamba" in name: 
        logger.info(f"Layer {name}: Shape {param.shape}")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Total trainable parameters: {total_params}\n")
logger.info(f"Model Summary: \n {model}\n")
logger.info("Training start.")
trainer.train(train_loader, val_loader, patience=config["early_stop"], model_folder=f"{result_path}/models/")
# used for visulization only.
test_original, test_reconstructed = trainer.evaluate(test_loader)
trainer.plot_results(test_original, test_reconstructed, trail_index=0, save_path = result_path, log=logger)