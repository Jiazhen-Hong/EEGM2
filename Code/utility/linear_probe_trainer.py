# linear_probe_trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, cohen_kappa_score
import os

from mamba_ssm import Mamba2 # JH download 11/14/2024, https://github.com/state-spaces/mamba (Complie with Numpy 1.x only, also required)

def train_mlp_for_linear_probe(
    train_features,    
    train_labels,      
    val_features,      
    val_labels,        
    test_features,    
    test_labels,      
    device,
    log_path,
    logger,
    num_classes=2,
    num_epochs_lp=20,
    batch_size_lp=64,
    lr_lp=1e-3
):

    # ---- 1) 
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

    # ---- 2)
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(SimpleMLP, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(p=0.5),   
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(p=0.5),   
                nn.Linear(32, num_classes)
            )
        def forward(self, x):
            return self.net(x)
        

    input_dim = train_features.shape[1]
    mlp_model = SimpleMLP(input_dim, num_classes).to(device)

    # ---- 3) 
    optimizer_lp = optim.AdamW(mlp_model.parameters(), lr=lr_lp)
    criterion_lp = nn.CrossEntropyLoss()

    logger.info(f"\n[MLP Linear-Probe] Start Training with: "
                f"epochs={num_epochs_lp}, batch_size={batch_size_lp}, lr={lr_lp}, num_classes={num_classes}")

    # ---- 4) Evaluation
    def evaluate_mlp(model, data_loader):
        model.eval()
        total_loss, correct, total_samples = 0.0, 0, 0
        all_probs, all_labels, all_preds = [], [], []
        with torch.no_grad():
            for xb, yb in data_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion_lp(out, yb)
                total_loss += loss.item() * xb.size(0)
                _, preds = torch.max(out, dim=1)
                correct += (preds == yb).sum().item()
                total_samples += xb.size(0)
                # softmax
                probs_full = nn.functional.softmax(out, dim=1)
                all_probs.append(probs_full.cpu())
                all_labels.append(yb.cpu())
                all_preds.append(preds.cpu())
        avg_loss = total_loss / total_samples
        acc = 100.0 * correct / total_samples
        all_probs_t = torch.cat(all_probs)  # shape: (N, num_classes)
        all_labels_t = torch.cat(all_labels)
        all_preds_t = torch.cat(all_preds)
        # AUROC 
        if len(torch.unique(all_labels_t)) == 2:
            auroc_val = roc_auc_score(all_labels_t.numpy(), all_probs_t.numpy()[:, 1])
        else:
            try:
                auroc_val = roc_auc_score(all_labels_t.numpy(), all_probs_t.numpy(), multi_class='ovr')
            except Exception as e:
                auroc_val = None
                logger.warning(f"AUROC计算失败：{e}")
        # others
        balanced_acc = balanced_accuracy_score(all_labels_t.numpy(), all_preds_t.numpy())
        weighted_f1 = f1_score(all_labels_t.numpy(), all_preds_t.numpy(), average='weighted')
        kappa = cohen_kappa_score(all_labels_t.numpy(), all_preds_t.numpy())
        return avg_loss, acc, auroc_val, balanced_acc, weighted_f1, kappa

    # ---- 5) 
    best_val_loss = float("inf")
    best_model_path = os.path.join(log_path, "best_linear_probe_mlp.pth")

    for epoch in range(num_epochs_lp):
        mlp_model.train()
        total_loss, correct, total_samples = 0.0, 0, 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer_lp.zero_grad()
            outputs = mlp_model(xb)
            loss = criterion_lp(outputs, yb)
            loss.backward()
            optimizer_lp.step()

            total_loss += loss.item() * xb.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == yb).sum().item()
            total_samples += xb.size(0)

        # train
        train_loss = total_loss / total_samples
        train_acc = 100.0 * correct / total_samples

        # validation
        val_loss, val_acc, val_auroc, val_balanced_acc, val_weighted_f1, val_kappa = evaluate_mlp(mlp_model, val_dl)

        # checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(mlp_model.state_dict(), best_model_path)
            logger.info(f"  [Epoch {epoch+1}] Validation loss improved to {val_loss:.4f}. Model saved.")

        logger.info(
            f"[Epoch {epoch+1}/{num_epochs_lp}] "
            f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.2f}%; "
            f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.2f}%, "
            f"ValAUROC={'{:.4f}'.format(val_auroc) if val_auroc is not None else 'N/A'}, "
            f"ValBalancedAcc={val_balanced_acc:.4f}, ValWeightedF1={val_weighted_f1:.4f}, "
            f"ValKappa={val_kappa:.4f}"
        )

    # ---- 6) load model
    logger.info(f"Loading best model from {best_model_path} for final test evaluation...")
    mlp_model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_auroc, test_balanced_acc, test_weighted_f1, test_kappa = evaluate_mlp(mlp_model, test_dl)
    logger.info(f"[Final Test] Loss={test_loss:.4f}, Acc={test_acc:.2f}%, "
                f"AUROC={'{:.4f}'.format(test_auroc) if test_auroc is not None else 'N/A'}, "
                f"BalancedAcc={test_balanced_acc:.4f}, WeightedF1={test_weighted_f1:.4f}, "
                f"Cohen's Kappa={test_kappa:.4f}")

    final_model_path = os.path.join(log_path, "final_linear_probe_mlp.pth")
    torch.save(mlp_model.state_dict(), final_model_path)
    logger.info(f"Final MLP checkpoint saved to {final_model_path}")
    logger.info("Finished MLP linear probing.\n")