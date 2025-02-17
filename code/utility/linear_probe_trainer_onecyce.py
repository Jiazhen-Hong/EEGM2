import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import OneCycleLR  
import os


def train_mlp_for_linear_probe(
    train_features,    # np.ndarray [N_train, D]
    train_labels,      # np.ndarray [N_train]
    val_features,      # np.ndarray [N_val, D]
    val_labels,        # np.ndarray [N_val]
    test_features,     # np.ndarray [N_test, D]
    test_labels,       # np.ndarray [N_test]
    device,
    log_path,
    logger,
    num_classes=2,
    num_epochs_lp=20,
    batch_size_lp=64,
    lr_lp=1e-3
):
    """
    用两层 MLP 做线性探针训练 + 验证集选最佳 + 测试集最终评估，并使用 OneCycleLR 学习率调度。

    Args:
        train_features (np.ndarray): [N_train, D], 训练集特征
        train_labels   (np.ndarray): [N_train], 训练集标签
        val_features   (np.ndarray): [N_val, D], 验证集特征
        val_labels     (np.ndarray): [N_val], 验证集标签
        test_features  (np.ndarray): [N_test, D], 测试集特征
        test_labels    (np.ndarray): [N_test], 测试集标签
        device (torch.device): 运行设备
        log_path (str): 日志/模型保存目录
        logger: 日志记录器
        num_classes (int): 分类数 (默认二分类=2)
        num_epochs_lp (int): 训练轮数
        batch_size_lp (int): batch size
        lr_lp (float): 学习率

    Returns:
        None
        （内部会打印并保存最优模型，最后在测试集评估）
    """

    # ---- 1) 封装 Dataset/DataLoader
    train_dataset = TensorDataset(
        torch.from_numpy(train_features).float(),
        torch.from_numpy(train_labels).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_features).float(),
        torch.from_numpy(val_labels).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_features).float(),
        torch.from_numpy(test_labels).long()
    )

    train_dl = DataLoader(train_dataset, batch_size=batch_size_lp, shuffle=True)
    val_dl   = DataLoader(val_dataset,   batch_size=batch_size_lp, shuffle=False)
    test_dl  = DataLoader(test_dataset,  batch_size=batch_size_lp, shuffle=False)

    # ---- 2) 定义两层 MLP
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(SimpleMLP, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        def forward(self, x):
            return self.net(x)

    input_dim = train_features.shape[1]
    mlp_model = SimpleMLP(input_dim, num_classes).to(device)

    # ---- 3) 定义损失和优化器 + OneCycleLR
    optimizer_lp = optim.AdamW(mlp_model.parameters(), lr=lr_lp)
    criterion_lp = nn.CrossEntropyLoss()

    # steps_per_epoch = 训练集的 iteration 数
    steps_per_epoch = len(train_dl)
    # total_steps = steps_per_epoch * epoch 数
    total_steps = steps_per_epoch * num_epochs_lp

    # 这里设置 max_lr = lr_lp，你可以根据需求将 max_lr 调得更高一些
    scheduler = OneCycleLR(
        optimizer_lp, 
        max_lr=lr_lp, 
        total_steps=total_steps,
        pct_start=0.1,        # 你可以根据需求调整 warm-up 区间
        anneal_strategy='cos' # 常用余弦退火
    )

    logger.info(
        f"\n[MLP Linear-Probe + OneCycleLR] Start Training with: "
        f"epochs={num_epochs_lp}, batch_size={batch_size_lp}, lr={lr_lp}, "
        f"num_classes={num_classes}, steps_per_epoch={steps_per_epoch}"
    )

    # ---- 4) 辅助函数：评估(返回loss,acc,auroc)，用于val/test
    def evaluate_mlp(model, data_loader):
        model.eval()
        total_loss, correct, total_samples = 0.0, 0, 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for xb, yb in data_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion_lp(out, yb)
                total_loss += loss.item() * xb.size(0)
                _, preds = torch.max(out, dim=1)
                correct += (preds == yb).sum().item()
                total_samples += xb.size(0)

                # 若二分类，取第1类的预测概率(softmax输出)
                if num_classes == 2:
                    probs = nn.functional.softmax(out, dim=1)[:, 1]
                    all_probs.append(probs.cpu())
                    all_labels.append(yb.cpu())

        avg_loss = total_loss / total_samples
        acc = 100.0 * correct / total_samples

        if num_classes == 2 and len(all_probs) > 0:
            all_probs_t = torch.cat(all_probs)
            all_labels_t = torch.cat(all_labels)
            auroc_val = roc_auc_score(all_labels_t.numpy(), all_probs_t.numpy())
        else:
            auroc_val = None
        return avg_loss, acc, auroc_val

    # ---- 5) 训练 + 验证集选最佳 (以最低 val_loss 作为最佳模型保存标准)
    best_val_loss = float("inf")
    best_model_path = os.path.join(log_path, "best_linear_probe_mlp.pth")

    for epoch in range(num_epochs_lp):
        mlp_model.train()
        total_loss, correct, total_samples = 0, 0, 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer_lp.zero_grad()
            outputs = mlp_model(xb)
            loss = criterion_lp(outputs, yb)
            loss.backward()
            optimizer_lp.step()

            # 更新 OneCycleLR
            scheduler.step()

            total_loss += loss.item() * xb.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == yb).sum().item()
            total_samples += xb.size(0)

        # 训练集表现
        train_loss = total_loss / total_samples
        train_acc = 100.0 * correct / total_samples

        # 验证集评估
        val_loss, val_acc, val_auroc = evaluate_mlp(mlp_model, val_dl)

        # 若验证集 loss 更低则保存checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(mlp_model.state_dict(), best_model_path)
            logger.info(f"  [Epoch {epoch+1}] Validation loss improved to {val_loss:.4f}. Model saved.")

        logger.info(
            f"[Epoch {epoch+1}/{num_epochs_lp}] "
            f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.2f}%, "
            f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.2f}%, "
            f"ValAUROC={val_auroc}"
        )

    # ---- 6) 载入最佳模型 + 测试集评估
    logger.info(f"Loading best model from {best_model_path} for final test...")
    mlp_model.load_state_dict(torch.load(best_model_path))

    test_loss, test_acc, test_auroc = evaluate_mlp(mlp_model, test_dl)
    logger.info(f"[Final Test] Loss={test_loss:.4f}, Acc={test_acc:.2f}%, AUROC={test_auroc}")

    # 也可把最终模型另存
    final_model_path = os.path.join(log_path, "final_linear_probe_mlp.pth")
    torch.save(mlp_model.state_dict(), final_model_path)
    logger.info(f"Final MLP checkpoint saved to {final_model_path}")
    logger.info("Finished MLP linear probing with OneCycleLR.\n")