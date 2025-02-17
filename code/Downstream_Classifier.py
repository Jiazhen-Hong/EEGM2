import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, cohen_kappa_score


class ClassificationTrainer:
    '''
    Trainer class for downstream classification tasks using a pre-trained model.
    '''
    def __init__(self, pretrained_model, in_features, num_classes, criterion, optimizer, num_epochs=10, num_layers_to_train = 0, log_dir=None, logger=None):
        # Build classifier
        self.model = self.build_classifier(pretrained_model, in_features, num_classes, num_layers_to_train, logger).to("cuda")
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        model_name = pretrained_model.__class__.__name__
        self.log_dir = log_dir or f'{model_name}_classification_{datetime.now().strftime("%Y%m%d-%H%M%S")}'

        # Initialize TensorBoard SummaryWriter
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)

    @staticmethod
    def set_fine_tuning(model, num_layers_to_train=0, logger=None):
        """
        Set fine-tuning for the model by unfreezing specified layers.

        Args:
            model (torch.nn.Module): Pretrained model to fine-tune.
            num_layers_to_train (int): 
                - 0: Freeze all layers (default behavior).
                - Positive integer: Unfreeze the last num_layers_to_train layers.
        """
        all_layers = list(model.named_parameters())
        num_total_layers = len(all_layers)
        if logger != None:
            logger.info(f"Total number of layers in the model: {num_total_layers}")

        if num_layers_to_train == 0:
            # Freeze all layers
            for name, param in all_layers:
                param.requires_grad = False
            if logger != None:
                logger.info("All layers are frozen.")
        else:
            # Freeze all layers initially
            for name, param in all_layers:
                param.requires_grad = False
            
            # Unfreeze the last num_layers_to_train layers
            for name, param in all_layers[-num_layers_to_train:]:
                param.requires_grad = True
                if logger != None:
                    logger.info(f"Unfrozen Layer: {name}")

    @staticmethod
    def build_classifier(pretrained_model, in_features, num_classes, num_layers_to_train=0, logger=None):
        '''
        Create a classifier by combining a pre-trained model (feature extractor) with a classification head.
        Allows partial fine-tuning by defining the number of trainable layers.
        '''
        class Classifier(nn.Module):
            def __init__(self, feature_extractor, in_features, num_classes):
                super(Classifier, self).__init__()
                self.feature_extractor = feature_extractor #pre-trained model, could compare
                self.fc = nn.Sequential(
                    nn.Linear(in_features, 128),  #  in_features = channels * times
                    nn.ReLU(),
                    nn.Linear(128, num_classes)
                )

            def forward(self, x):
                features = self.feature_extractor(x)
                features = features.view(features.size(0), -1)
                out = self.fc(features)
                return out

        # 应用部分 fine-tuning
        ClassificationTrainer.set_fine_tuning(pretrained_model, num_layers_to_train, logger=logger)

        return Classifier(pretrained_model, in_features, num_classes)

    def train(self, train_loader, val_loader, patience=5, checkpoint_path="checkpoint.pth", logger=None):
        '''
        Train the classification model with validation, log memory usage, and early stopping.
        '''
        early_stopping = EarlyStopping(patience=patience, save_path=checkpoint_path)
        self.model.train()
        total_training_time = 0.0  # Accumulate training time

        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            running_loss = 0.0
            correct = 0
            total = 0

            # Training phase
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Log loss to TensorBoard every 10 batches
                # if batch_idx % 10 == 0:
                #     self.writer.add_scalar('Loss/train-steps', loss.item(), epoch * len(train_loader) + batch_idx)

            # Compute epoch metrics
            epoch_duration = time.time() - epoch_start_time
            total_training_time += epoch_duration
            avg_loss = running_loss / len(train_loader)
            accuracy = 100 * correct / total

            # Memory usage
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
            max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB

            # Validation phase
            val_loss, val_accuracy = self.validate(val_loader)
            
            # 清理 CUDA 缓存
            torch.cuda.empty_cache()

            # Early stopping check
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

            # Print and log
            logger.info(f"Epoch [{epoch + 1}/{self.num_epochs}], "
                f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, "
                f"Duration: {epoch_duration:.2f}s, "
                f"Memory Allocated: {memory_allocated:.2f} MB, Max Memory: {max_memory_allocated:.2f} MB")
            
            self.writer.add_scalar('Loss/train-epoch', avg_loss, epoch) 
            self.writer.add_scalar('Accuracy/train', accuracy, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            self.writer.add_scalar('Time/epoch_duration', epoch_duration, epoch)
            self.writer.add_scalar('Memory/memory_allocated', memory_allocated, epoch)
            self.writer.add_scalar('Memory/max_memory_allocated', max_memory_allocated, epoch)

        # Log average training time
        average_training_time = total_training_time / (epoch + 1)  # Use the actual number of epochs
        logger.info(f"Average Training Time per Epoch: {average_training_time:.2f} seconds")
        self.writer.add_text('Training Summary', f'Average Training Time per Epoch: {average_training_time:.2f} seconds')
        self.writer.close()

        # 加载保存的最佳模型
        self.model.load_state_dict(torch.load(checkpoint_path))
        logger.info(f"Loaded the best model from {checkpoint_path} with validation loss {early_stopping.best_loss:.4f}")
        return self.model

    def validate(self, val_loader):
        '''
        Validate the model on the validation dataset.
        '''
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to("cuda"), labels.to("cuda")

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(val_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def evaluate(self, test_loader, logger=None):
        '''
        Evaluate the classification model on the test dataset, calculating accuracy and AUROC for binary classification.
        '''
        self.model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
                outputs = self.model(inputs)
                
                # Predictions and probabilities
                _, predicted = torch.max(outputs, 1)
                probs = F.softmax(outputs, dim=1)[:, 1]  # Probabilities for the positive class
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Collect labels and probabilities for AUROC
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())

        # Calculate accuracy
        accuracy = 100 * correct / total

        # Concatenate all labels and probabilities
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()

        # Ensure binary classification for AUROC
        if len(set(all_labels)) == 2:  # Only calculate AUROC if both classes are present
            auroc = roc_auc_score(all_labels, all_probs)
            logger.info(f"AUROC on the test set: {auroc:.4f}")
        else:
            raise ValueError("AUROC can only be calculated for binary classification (exactly two classes in the test set).")

        # Log accuracy
        logger.info(f"Accuracy on the test set: {accuracy:.2f}%")
        return accuracy, auroc


###### balance accurary

    def evaluate(self, test_loader, logger=None):
        '''
        Evaluate the classification model on the test dataset, calculating accuracy, AUROC, and Balanced Accuracy.
        '''
        self.model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
                outputs = self.model(inputs)
                
                # Predictions and probabilities
                _, predicted = torch.max(outputs, 1)
                probs = F.softmax(outputs, dim=1)[:, 1]  # Probabilities for the positive class
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Collect labels and predictions for Balanced Accuracy
                all_labels.append(labels.cpu())
                all_predictions.append(predicted.cpu())
                
                # Collect probabilities for AUROC
                all_probs.append(probs.cpu())

        # Calculate accuracy
        accuracy = 100 * correct / total

        # Concatenate all labels and predictions
        all_labels = torch.cat(all_labels).numpy()
        all_predictions = torch.cat(all_predictions).numpy()

        # Calculate Balanced Accuracy
        balanced_acc = balanced_accuracy_score(all_labels, all_predictions)

        # Concatenate probabilities for AUROC
        all_probs = torch.cat(all_probs).numpy()

        # Ensure binary classification for AUROC
        if len(set(all_labels)) == 2:  # Only calculate AUROC if both classes are present
            auroc = roc_auc_score(all_labels, all_probs)
            logger.info(f"AUROC on the test set: {auroc:.4f}")
        else:
            raise ValueError("AUROC can only be calculated for binary classification (exactly two classes in the test set).")

        # Log metrics
        logger.info(f"Accuracy on the test set: {accuracy:.2f}%")
        logger.info(f"Balanced Accuracy on the test set: {balanced_acc:.4f}")
        return accuracy, auroc, balanced_acc


    ###### weighted F1
    def evaluate(self, test_loader, logger=None):
        '''
        Evaluate the classification model on the test dataset, calculating:
        - Accuracy
        - AUROC (for multi-class)
        - Balanced Accuracy
        - Weighted F1 Score
        - Cohen's Kappa
        '''
        self.model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
                outputs = self.model(inputs)
                
                # Predictions and probabilities
                _, predicted = torch.max(outputs, 1)  # 多分类取最大值对应类别
                probs = F.softmax(outputs, dim=1)  # 获取所有类别的概率分布
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Collect labels and predictions for further metrics
                all_labels.append(labels.cpu())
                all_predictions.append(predicted.cpu())
                all_probs.append(probs.cpu())

        # **计算 Accuracy**
        accuracy = 100 * correct / total

        # **将数据转换为 NumPy 格式**
        all_labels = torch.cat(all_labels).numpy()
        all_predictions = torch.cat(all_predictions).numpy()
        all_probs = torch.cat(all_probs).numpy()

        # **计算 Balanced Accuracy**
        balanced_acc = balanced_accuracy_score(all_labels, all_predictions)

        # **计算 Weighted F1 Score**
        weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')

        # **计算 Cohen's Kappa**
        cohen_kappa = cohen_kappa_score(all_labels, all_predictions)

        # **计算 AUROC（适用于多分类）**
        unique_classes = len(set(all_labels.tolist()))
        if unique_classes == 2:  # **二分类**
            auroc = roc_auc_score(all_labels, all_probs[:, 1])  # 仅取正类别的概率
            logger.info(f"Binary AUROC on the test set: {auroc:.4f}")
        elif unique_classes > 2:  # **多分类**
            auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr')  # One-vs-Rest
            logger.info(f"Multi-class AUROC on the test set: {auroc:.4f}")
        else:
            auroc = None
            logger.warning("AUROC cannot be computed: Less than two classes detected.")

        # **日志记录**
        logger.info(f"Accuracy on the test set: {accuracy:.2f}%")
        logger.info(f"Balanced Accuracy on the test set: {balanced_acc:.4f}")
        logger.info(f"Weighted F1 Score on the test set: {weighted_f1:.4f}")
        logger.info(f"Cohen's Kappa on the test set: {cohen_kappa:.4f}")

        return accuracy, auroc, balanced_acc, weighted_f1, cohen_kappa


class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, save_path="checkpoint.pth"):
        """
        Args:
            patience (int): 连续多少个 epoch 没有改善后停止训练。
            delta (float): 最小改善幅度。
            save_path (str): 保存模型的路径。
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        # 检查是否有改善
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """保存当前最佳模型"""
        torch.save(model.state_dict(), self.save_path)
        print(f"Validation loss improved, model saved to {self.save_path}")