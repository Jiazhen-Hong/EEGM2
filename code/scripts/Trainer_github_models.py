import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import json
from Github_models.EEG2Rep_models.models import EEG2Rep  # 导入EEG2Rep模型
from utility import chebyBandpassFilter, setup_logging, create_data_loaders
from sklearn.metrics import roc_auc_score  # 用于计算AUROC
#from Github_models.BENDR.reproduction_JH_selfTransformer import BENDR
from Github_models.MAEEG import *
from Github_models.Trainer_MAEEG_JH import *
from Github_models.BIOT_models.biot import UnsupervisedPretrain, BIOTClassifier
from Github_models.BENDR.Trainer_BENDR import BENDR, BENDRTrainer


# -------------------------------
# Configuration and Logging Setup   
# -------------------------------
data_name = 'Alpha'  # Alpha, Attention, Crowdsource, STEW, DriverDistraction, DREAMER, TUAB, TUEV
model_name = "BENDRTrainer"   # EEG2Rep, BIOT, MAEEG, MAEEGTrainer, BENDRTrainer

# Load json for hyper-parameter
with open("config.json", "r") as f:
    config = json.load(f)

# Define result path & log path
result_path = f'../results_paper/{data_name}/{datetime.now().strftime("%Y%m%d")}/{model_name}-{datetime.now().strftime("%H%M%S")}/'
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
    logger.info("=" * 80)
    logger.info(f"Successful Load the Dataset: {data_name}")
    logger.info("=" * 80)
    train_data, train_label, test_data, test_label, val_data, val_label = \
        data_x_dir['train_data'], data_x_dir['train_label'], \
        data_x_dir['test_data'], data_x_dir['test_label'], \
        data_x_dir['val_data'], data_x_dir['val_label']

    logger.info(f"  - Train data shape: {train_data.shape}, Train label shape: {train_label.shape}")
    logger.info(f"  - Validation data shape: {val_data.shape}, Validation label shape: {val_label.shape}")
    logger.info(f"  - Test data shape: {test_data.shape}, Test label shape: {test_label.shape}")
    
    # Applying chebyBandpassFilter
    if config["filter"]:
        train_data = chebyBandpassFilter(train_data, [0.2, 0.5, 40, 48], gstop=40, gpass=1, fs=128)
        val_data = chebyBandpassFilter(val_data, [0.2, 0.5, 40, 48], gstop=40, gpass=1, fs=128)
        test_data = chebyBandpassFilter(test_data, [0.2, 0.5, 40, 48], gstop=40, gpass=1, fs=128)
        logger.info('Filtered all datasets.')

    # Dataset loader and info print
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data=train_data,
        train_label=train_label,
        val_data=val_data,
        val_label=val_label,
        test_data=test_data,
        test_label=test_label,
        batch_size=config["batch_size"]
    )
else:
    raise ValueError(f"Unknown data_name: {data_name}")

# Print dataset info
data_batch = next(iter(train_loader))
logger.info(f"Shape for each batch: {data_batch[0].shape}")
in_channels = data_batch[0].shape[1]
seq_len = data_batch[0].shape[2]

###########################
# --   Model Setup    -- 
###########################
learningrate = 0.001

if model_name == "EEG2Rep":
    # 读取 EEG2Rep 的配置
    with open("./Github_models/EEG2Rep_models/config_eeg2rep.json", "r") as f:
        config_eeg2rep = json.load(f)

    # 更新配置文件中的数据形状
    config_eeg2rep['Data_shape'] = [config["batch_size"], in_channels, seq_len]
    learningrate = config_eeg2rep['lr']
    model = EEG2Rep(config_eeg2rep, num_classes=len(np.unique(train_label))).to(device)


elif model_name == "BIOT":
    model = BIOTClassifier(
            emb_size=256,
            heads=8,
            depth=4,
            n_classes=len(np.unique(train_label)),
            n_fft=200,
            hop_length=100
        ).to(device)
    
elif model_name == "MAEEG":
    # 初始化 MAEEGClassification 使用默认参数
    model = MAEEGClassification(
        input_channel=14,   # 默认 EEG 通道数
        embed_size=128,     # 默认嵌入维度
        downsample_size=[2, 2, 2],  # 默认下采样
        kernel_size=[3, 3, 3],      # 默认卷积核大小
        dropout=0.1,        # 默认 Dropout 概率
        transformer_embed_size=128, # 默认 Transformer 嵌入维度
        heads=8,            # 默认 Transformer 头数
        forward_neuron=256, # 默认前馈层神经元数
        num_transformers=4, # 默认 Transformer 层数
        num_classes=len(np.unique(train_label))  # 根据训练集标签确定类别数
    ).to(device)
elif model_name == "MAEEGTrainer":
    trainer = MAEEGTrainer(train_loader, val_loader, test_loader, device, result_path, config['num_epochs'])
    
    trainer.pretrain()  # 第一阶段：预训练
    trainer.finetune()  # 第二阶段：分类微调
    trainer.test()      # 测试模型
    exit()
elif model_name == "BENDR":
    model = BENDR(
        in_channels=14,  # EEG 通道数
        encoder_dim=512,  # Encoder 维度
        num_layers=8,  # Transformer 层数
        num_heads=8,  # 注意力头数
        ff_dim=3076,  # 前馈层维度
        num_classes=2  # 二分类
    ).to(device)
elif model_name == "BENDRTrainer":
    trainer = BENDRTrainer(train_loader, val_loader, test_loader, device, result_path, config['num_epochs'])
    
    trainer.pretrain()  # 第一阶段：预训练
    trainer.finetune()  # 第二阶段：分类微调
    trainer.test()      # 测试模型
    exit()

else:
    raise ValueError(f"Model name {model_name} not recognized.")

###########################
# --   Training Loop   -- 
###########################

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learningrate)

# best_model_path = os.path.join(result_path, "best_model.pth")
# best_val_accuracy = 0.0  # 保存验证集最高准确率

# for epoch in range(config['num_epochs']):
#     model.train()
#     running_loss = 0.0
#     for data, labels in train_loader:
#         data, labels = data.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(data)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     logger.info(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {running_loss / len(train_loader)}")

#     # Validation
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data, labels in val_loader:
#             data, labels = data.to(device), labels.to(device)
#             outputs = model(data)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     val_accuracy = 100 * correct / total
#     logger.info(f"Validation Accuracy after epoch {epoch + 1}: {val_accuracy:.2f}%")

#     # 保存最佳模型
#     if val_accuracy > best_val_accuracy:
#         best_val_accuracy = val_accuracy
#         torch.save(model.state_dict(), best_model_path)
#         logger.info(f"Saved best model with Validation Accuracy: {best_val_accuracy:.2f}%")

# logger.info("Training completed.")
# logger.info(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")

best_val_loss = float('inf')  # 记录最优的 Validation Loss
best_model_path = os.path.join(result_path, "best_model.pth")

for epoch in range(config['num_epochs']):
    model.train()
    running_loss = 0.0
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    logger.info(f"Epoch [{epoch + 1}/{config['num_epochs']}], Training Loss: {avg_train_loss:.4f}")

    # **Validation**
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    logger.info(f"Validation Loss after epoch {epoch + 1}: {avg_val_loss:.4f}")

    # **保存最佳模型（最小 Loss）**
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        logger.info(f"Saved best model with Validation Loss: {best_val_loss:.4f}")

logger.info("Training completed.")
logger.info(f"Best Validation Loss: {best_val_loss:.4f}")


###########################
# --  Test Set Metrics  -- 
###########################

model.eval()
correct = 0
total = 0
all_labels = []
all_probs = []

with torch.no_grad():
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # 假设是二分类任务
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

accuracy = 100 * correct / total
auroc = roc_auc_score(all_labels, all_probs)

logger.info(f"Test Set Accuracy: {accuracy:.2f}%")
logger.info(f"Test Set AUROC: {auroc:.4f}")
print(f"Test Set Accuracy: {accuracy:.2f}%")
print(f"Test Set AUROC: {auroc:.4f}")