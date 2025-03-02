from Downstream_Classifier import *
#from Trainer import *
from models import EEGM2, EEGM2_S1, EEGM2_S3, EEGM2_S4, EEGM2_S5
import torch.optim as optim
from datetime import datetime
import json
from utility import setup_logging, TUABLoader
from utility.data_loader_filter import chebyBandpassFilter, create_data_loaders
from utility.custerm_logistic_regression import fit_lr
from utility.NL_representation import extract_pooled_representation, count_parameters_up_to
from collections import Counter
from sklearn.metrics import accuracy_score

import os
import joblib
import numpy as np


# Load json for hyper-parameter
with open("config_downstream.json", "r") as f:
    config = json.load(f)

data_name = 'TUAB' #Alpha, Attention, Crowdsource, STEW, DriverDistraction, : DREAMER, TUAB
model_name = "EEGM2"
date = "100s-L1Spectral"
downstream_data_name = 'TUAB' #'STEW', 'Crowdsource', 'Attention', 'Alpha', 'DriverDistraction', 'DREAMER', 'TUAB', 


model_on_single_GPU = True
if "cuda:0" in config["GPU_device"]:
    torch.cuda.set_per_process_memory_fraction(0.9, device=0) #0.35
elif "cuda:1" in config["GPU_device"]:
    torch.cuda.set_per_process_memory_fraction(0.9, device=1)
device = torch.device(config["GPU_device"])

model_location = f'{date}/{model_name}' 

##############################
#       Seed
##############################
import random
seed = config["Seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


target_layer = config["target_layer"]
#bottleneck, pool4, encoder4


##############################
#       Initial Train
##############################
# check the downstream task type.
if config["downstream_task_type"] == 1:
    #Fine-tuned Classifier
    save_path = f'1_fine_{config["num_layers_to_train"]}'
    print(f"Task: Fine-tuned Classifier, save path: {save_path}")
elif config["downstream_task_type"] == 2:
    #Classifier from Scratch
    save_path = '2_scratch'
    print(f"Task: Classifier from Scratch, save path: {save_path}")
elif config["downstream_task_type"] == 3:
    #Linear Probing
    save_path = f'3_Linear_{target_layer}'
    print(f"Task: Linear Probing, save path: {save_path}")

else:
    raise ValueError("Invalid downstream_task_type. Must be 1 (fune), 2 (scratch), or 3 (linear).")


result_path =f"../results_paper/{data_name}/{model_location}"

# logger
if downstream_data_name == data_name:
    log_path = f"{result_path}/downstrem{save_path}/seed_{seed}/"
else:
    log_path = f"{result_path}/{downstream_data_name}/downstrem{save_path}/seed_{seed}/"


if not os.path.exists(log_path):
    os.makedirs(log_path)
logger, log_file = setup_logging(log_path)
logger.info(f"Loaded hyperparameters: [{config}]. \n")


##############################
#       Data Loading
##############################

if downstream_data_name == 'TUAB':
    if date == "10s-L1Spectral":
        tuab_root = "/data/datasets_public/TUAB/edf/processed_128hz_1280seqlen_JH/"
    elif date == "30s-L1Spectral":
        tuab_root = "/data/datasets_public/TUAB/edf/processed_128hz_3840seqlen_JH/"
    elif date == "60s-L1Spectral" or date == "60s-L2Spectral":
        tuab_root = "/data/datasets_public/TUAB/edf/processed_128hz_7680seqlen_JH/"
    elif date == "100s-L1Spectral" or date == "100s-L1":
        tuab_root = "/data/datasets_public/TUAB/edf/processed_128hz_12800seqlen_JH/"
    elif date == "2s-L1Spectral":
        tuab_root = "/data/datasets_public/TUAB/edf/processed_128hz_256seqlen/"
    else:
        print('data dimension error')
        exit()

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
    for _, labels in train_loader:
        print(f"Labels min: {labels.min()}, max: {labels.max()}")
        break  

    logger.info("Loaded TUAB dataset.")
else:
    data_x_dir = np.load(f'/data/data_downstream_task/{downstream_data_name}/{downstream_data_name}.npy', allow_pickle=True).item()
    logger.info(f"\n Successful Load the Dataset: {downstream_data_name} \n")
    train_data, train_label, test_data, test_label, val_data, val_label, All_train_data, All_train_label = \
        data_x_dir['train_data'], data_x_dir['train_label'], data_x_dir['test_data'], data_x_dir['test_label'], \
        data_x_dir['val_data'], data_x_dir['val_label'], data_x_dir['All_train_data'], data_x_dir['All_train_label']
    logger.info(f"  - Train data shape: {train_data.shape}, Train label shape: {train_label.shape}")
    logger.info(f"  - Test data shape: {test_data.shape}, Test label shape: {test_label.shape}")
    logger.info(f"  - Validation data shape: {val_data.shape}, Validation label shape: {val_label.shape}")
    logger.info(f"  - All train data shape: {All_train_data.shape}, All train label shape: {All_train_label.shape}\n")
    total_trails = All_train_data.shape[0] + test_data.shape[0]
    logger.info(f"Total number of trails: {total_trails}")


    # Applying chebyBandpassFilter
    if config["filter"]:
        train_data = chebyBandpassFilter(train_data, [0.2, 0.5, 40, 48], gstop=40, gpass=1, fs=128)
        logger.info('training set filtered.')
        val_data = chebyBandpassFilter(val_data, [0.2, 0.5, 40, 48], gstop=40, gpass=1, fs=128)
        logger.info('validation set filtered.')
        test_data = chebyBandpassFilter(test_data, [0.2, 0.5, 40, 48], gstop=40, gpass=1, fs=128)
        logger.info('testing set filtered.')

    train_loader, val_loader, test_loader = create_data_loaders(
        train_data=train_data,
        train_label=train_label,
        val_data=val_data,
        val_label=val_label,
        test_data=test_data,
        test_label=test_label,
        batch_size=config["batch_size"]
    )
for name, loader in [("Training", train_loader), ("Validation", val_loader), ("Testing", test_loader)]:
    num_batches, total_trails = len(loader), len(loader.dataset)
    logger.info(f"{name} set: Number of batches: {num_batches}")
    logger.info(f"{name} set: Total number of trails: {total_trails}\n")
data_batch = next(iter(train_loader))
logger.info(f"Shape for each batch: {data_batch[0].shape}")
in_times, in_channels = data_batch[0].shape[2], data_batch[0].shape[1]
logger.info(f"    # Duration (s): {in_times}\n    # of channels: {in_channels}")


#######################################
#      Training: Classification Task
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
else:
    raise ValueError("model name does not exist.")


if config["downstream_task_type"] == 1:
    # Fine-tuned Classifier
    logger.info(f"Using pretrained model for fine-tuning, Unfroze last [{config['num_layers_to_train']}] layers.")
    if not model_on_single_GPU:
        model = model.to(device)
        model = torch.nn.DataParallel(model)  
    else:
        model = model.to(device)  
    checkpoint = torch.load(f"{result_path}/models/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint)  
    if config["num_layers_to_train"] == 0:
        model.eval() 


elif config["downstream_task_type"] == 2:
    # Classifier from Scratch
    logger.info("Using model from scratch for classification.")
    model = model.to(device)

elif config["downstream_task_type"] == 3:
    # Linear Probing
    logger.info("Using pretrained model for linear probing.")
    model = model.to(device)
    if not model_on_single_GPU:
        model = model.to(device)
        model = torch.nn.DataParallel(model)  
    else:
        model = model.to(device)  
    checkpoint = torch.load(f"{result_path}/models/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    params_count = count_parameters_up_to(model, target_layer)
    logger.info(f"Trainable parameters up to {target_layer}: {params_count}")
    logger.info("Model structure and module names:")
    for name, module in model.named_modules():
        logger.info(f"Module name: {name}, Module type: {type(module)}")
    logger.info("Model structure printed successfully.")
   

    # ---------------------------
    #  Step 1: make representation
    # ---------------------------
    train_features, train_labels = extract_pooled_representation(
        model=model,
        loader=train_loader,
        target_submodule_name=target_layer,  # 比如 "bottleneck"
        device=device
    )
    logger.info(f"Train feature representation done. Feature shape: {train_features.shape}, Labels shape: {train_labels.shape}")
    # test
    test_features, test_labels = extract_pooled_representation(
        model=model,
        loader=test_loader,
        target_submodule_name=target_layer,
        device=device
    )
    logger.info(f"Train feature representation done. Feature shape: {test_features.shape}, Labels shape: {test_labels.shape}")
    #val
    val_features, val_labels = extract_pooled_representation(
    model=model,
    loader=val_loader,
    target_submodule_name=target_layer,  # 比如 "bottleneck"
    device=device
    )
    logger.info(f"Train feature representation done. Feature shape: {val_features.shape}, Labels shape: {val_labels.shape}")
    

    train_features = train_features.cpu().numpy()
    train_labels   = train_labels.cpu().numpy()
    test_features  = test_features.cpu().numpy()
    test_labels    = test_labels.cpu().numpy()
    val_features  = val_features.cpu().numpy()
    val_labels    = val_labels.cpu().numpy()
    

    if config["Linear_Prbing"] is True:

        classifier = fit_lr(train_features, train_labels, seed)
        classifier_save_path = f"{log_path}/linear_probe_model.pkl"
        joblib.dump(classifier, classifier_save_path)
        
        test_predictions = classifier.predict(test_features)
        test_probabilities = classifier.predict_proba(test_features)

        accuracy = accuracy_score(test_labels, test_predictions)
        logger.info(f"Linear Probing Accuracy: {accuracy:.4f}")

        # Balanced Accuracy
        balanced_acc = balanced_accuracy_score(test_labels, test_predictions)
        logger.info(f"Linear Probing Balanced Accuracy: {balanced_acc:.4f}")

        # Weighted F1 Score
        weighted_f1 = f1_score(test_labels, test_predictions, average='weighted')
        logger.info(f"Linear Probing Weighted F1 Score: {weighted_f1:.4f}")

        # Cohen's Kappa
        cohen_kappa = cohen_kappa_score(test_labels, test_predictions)
        logger.info(f"Linear Probing Cohen's Kappa: {cohen_kappa:.4f}")

        # AUROC
        unique_classes = len(set(test_labels.tolist()))
        if unique_classes == 2:  # **binary**
            auroc = roc_auc_score(test_labels, test_probabilities[:, 1])
            logger.info(f"Linear Probing Binary AUROC: {auroc:.4f}")
        elif unique_classes > 2:  # **multi**
            auroc = roc_auc_score(test_labels, test_probabilities, multi_class='ovr')
            logger.info(f"Linear Probing Multi-class AUROC: {auroc:.4f}")
        else:
            auroc = None
            logger.warning("Cannot compute AUROC: Less than two classes detected.")

    else:
        from utility.linear_probe_trainer import train_mlp_for_linear_probe
        train_mlp_for_linear_probe(
            train_features=train_features,  # np array
            train_labels=train_labels,      # np array
            val_features=val_features,      # np array
            val_labels=val_labels,          # np array
            test_features=test_features,    # np array
            test_labels=test_labels,        # np array
            device=device,
            log_path=log_path,
            logger=logger,
            num_classes=config["num_classes"],   
            num_epochs_lp=config['num_epochs'],
            batch_size_lp=config['batch_size'],
            lr_lp=config['learning_rate']
        )



    exit()  # Linear Probing does not require further steps
    
else:
    raise ValueError("Invalid downstream task type.")



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"]) #, SGD, optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
classifier_trainer = ClassificationTrainer(model, int(in_channels*in_times), config["num_classes"], criterion, optimizer, num_epochs=config["num_epochs"], num_layers_to_train=config["num_layers_to_train"], log_dir=log_path, logger=logger)
# Train the classifier
logger.info("Training start.")
best_model = classifier_trainer.train(train_loader, val_loader, patience=config["early_stop"], checkpoint_path=f"{log_path}checkpoint.pth", logger=logger)
logger.info("Training end.")
# Evaluate the classifier

accuracy = classifier_trainer.evaluate(test_loader, logger=logger)

# Save the trained classifier
torch.save(classifier_trainer.model.state_dict(), f"{log_path}classifier_model.pth")
logger.info(f"Classifier model saved to {log_path}")





