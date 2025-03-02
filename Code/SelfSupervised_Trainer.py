"""
Created on Mon Nov 15 10:12:39 2024

@author: jiazhen@emotiv.com
"""
import torch
print(torch.__version__)
print(torch.version.cuda)
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
from numpy.linalg import norm
import time  
import torch.cuda  
from torch.optim.lr_scheduler import OneCycleLR
class Trainer:
    '''
    Model training for Self-supervised Reconstruction Task.
    '''
    def __init__(self, model, criterion, optimizer, num_epochs=10, log_dir=None):
        self.model = model.to("cuda")
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        model_name = model.__class__.__name__
        self.log_dir = log_dir or f'{model_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        # Initialize TensorBoard SummaryWriter
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
    @staticmethod
    def print_progress_bar(current, total, bar_length=50):
        fraction = current / total
        filled_length = int(fraction * bar_length)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        percent = fraction * 100
        print(f'\rProgress: |{bar}| {percent:.1f}%', end='', flush=True)
        if current == total:
            print()  # Move to the next line at the end of the progress bar
    @staticmethod
    def save_plot(filename, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plot_filename = os.path.join(save_path, f'{filename}.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f"Successfully saved to {plot_filename}")
    def train(self, train_loader, val_loader=None, patience=10, model_folder = None , checkpoint_path="checkpoint_epoch_{epoch}.pth", best_model_path="best_model.pth"):
        # Initialize EarlyStopping
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        early_stopping = EarlyStopping(
            patience=patience, 
            save_path= model_folder+best_model_path, 
            checkpoint_path = model_folder+checkpoint_path
        )
        # Initial OneCycle learning rate
        steps_per_epoch = len(train_loader)  
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=5e-4,                     
            epochs=self.num_epochs,          
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,                   
            anneal_strategy='cos',          
            div_factor=2,                   
            final_div_factor=100             
        )
        self.model.train()
        total_training_time = 0.0  # Accumulate training time
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            running_loss = 0.0

            # Training phase
            self.print_progress_bar(0, len(train_loader))
            for batch_idx, batch in enumerate(train_loader):
                inputs = batch[0].to("cuda")
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)  # Compute reconstruction loss
                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                scheduler.step()  # update learning rate
                running_loss += loss.item()
                if batch_idx % 100 == 0:  
                    self.print_progress_bar(batch_idx + 1, len(train_loader))

            # Compute epoch statistics
            epoch_duration = time.time() - epoch_start_time
            total_training_time += epoch_duration
            avg_loss = running_loss / len(train_loader)
            # Validation phase
            if val_loader:
                val_loss = self.validate(val_loader)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                # Save checkpoints and check early stopping
                stop_training = early_stopping(val_loss, self.model, epoch)
            else:
                val_loss = None
                stop_training = False
            # Memory usage
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
            max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
            # Print and log epoch results
            val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], "
                f"Train Loss: {avg_loss:.4f}, "
                f"Val Loss: {val_loss_str}, "
                f"Duration: {epoch_duration:.2f}s, "
                f"Memory Allocated: {memory_allocated:.2f} MB, Max Memory: {max_memory_allocated:.2f} MB")
            self.writer.add_scalar('Loss/train-epoch', avg_loss, epoch)
            self.writer.add_scalar('Time/epoch_duration', epoch_duration, epoch)
            self.writer.add_scalar('Memory/memory_allocated', memory_allocated, epoch)
            self.writer.add_scalar('Memory/max_memory_allocated', max_memory_allocated, epoch)
            # Stop training if patience is exceeded
            if stop_training:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break
        # Log average training time
        average_training_time = total_training_time / self.num_epochs
        print(f"Average Training Time per Epoch: {average_training_time:.2f} seconds")
        self.writer.add_text('Training Summary', f'Average Training Time per Epoch: {average_training_time:.2f} seconds')
        self.writer.close()
    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to("cuda")
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                running_loss += loss.item()
        avg_loss = running_loss / len(val_loader)
        return avg_loss
    def evaluate(self, test_loader):
        self.model.eval()
        test_reconstructed = []
        test_original = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch[0].to("cuda")
                outputs = self.model(inputs).cpu().numpy()
                test_reconstructed.append(outputs)
                test_original.append(inputs.cpu().numpy())
            test_reconstructed = np.concatenate(test_reconstructed, axis=0)
            test_original = np.concatenate(test_original, axis=0)
        return test_original, test_reconstructed
        
    def plot_results(self, test_original, test_reconstructed, trail_index=0, save_path=None, log=None):
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
        
        original_mean = np.mean(test_original, axis=0)  # shape: (time_points, channels)
        reconstructed_mean = np.mean(test_reconstructed, axis=0)  # shape: (time_points, channels)
        # MSE
        total_mse = 0
        for channel in range(original_mean.shape[0]):
            mse = np.mean((original_mean[channel, :] - reconstructed_mean[channel, :]) ** 2)
            total_mse += mse
            log.info(f" - Channel {channel + 1} MSE: {mse}")
        average_mse = total_mse / original_mean.shape[1]
        log.info(f' - Average MSE across all channels computed from Original and Reconstructed ERPs: {average_mse}')
        # Plot 1
        plt.figure(figsize=(14, 6))
        for channel in range(1):
            plt.plot(test_original[trail_index, channel, :], label=f'Original Channel {channel + 1}', color='black', alpha=0.3)
            plt.plot(test_reconstructed[trail_index, channel, :], label=f'Reconstructed Channel {channel + 1}', color='red', alpha=0.3)
        plt.title(f'Original and Reconstructed Signals (Trail={trail_index}, Channel=1)')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.tight_layout()
        if save_path:
            self.save_plot('plot1', save_path)
        plt.show()
        # Plot 2
        plt.figure(figsize=(14, 6))
        for channel in range(1):
            plt.plot(original_mean[channel, :], label=f'Original Channel {channel + 1}', color='black', alpha=0.3)
            plt.plot(reconstructed_mean[channel, :], label=f'Reconstructed Channel {channel + 1}', color='red', alpha=0.3)
        plt.title('Original and Reconstructed Signals (Average Across Trials, Channel=1)')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.tight_layout()
        if save_path:
            self.save_plot('plot2', save_path)
        plt.show()
        # Plot 3
        plt.figure(figsize=(14, 6))
        for channel in range(original_mean.shape[0]):
            plt.plot(original_mean[channel, :], label=f'Original Channel {channel + 1}', color='black', alpha=0.3)
            plt.plot(reconstructed_mean[channel, :], label=f'Reconstructed Channel {channel + 1}', color='red', alpha=0.3)
        plt.title('Original and Reconstructed Signals (Average Across Trials, All Channels)')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.tight_layout()
        if save_path:
            self.save_plot('plot3', save_path)
        plt.show()
        # Plot 4
        original_Potential = np.mean(test_original, axis=(1, 2)) 
        reconstructed_Potential = np.mean(test_reconstructed, axis=(1, 2))  
        plt.figure(figsize=(14, 6))
        plt.plot(original_Potential, label='Original Channel', color='black', alpha=0.3)
        plt.plot(reconstructed_Potential, label='Reconstructed Channel', color='red', alpha=0.3)
        plt.title('Original and Reconstructed Signals (Average Across Trials, Average Across Channels)')
        plt.xlabel('Time Points')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.tight_layout()
        if save_path:
            self.save_plot('plot4', save_path)
        plt.show()
class EarlyStopping:
    def __init__(self, patience=10, save_path="best_model.pth", checkpoint_path="checkpoint_epoch_{epoch}.pth"):
        self.patience = patience
        self.save_path = save_path
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")
        self.counter = 0  # Tracks epochs without improvement
        self.best_model_state = None  # To store the best model state
        self.checkpoint_files = []  # List to track saved checkpoints
    def __call__(self, val_loss, model, epoch):
        # Check if validation loss has improved
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()  # Save the best model's state
            self.save_checkpoint(model, epoch)
            self.save_best_model(model)
        else:
            self.counter += 1
            print(f"No improvement for {self.counter} epoch(s).")
        return self.counter >= self.patience  # Stop training if patience is exceeded
    def save_checkpoint(self, model, epoch):
        """Save the current model as a checkpoint and manage checkpoint files."""
        checkpoint_file = self.checkpoint_path.format(epoch=epoch + 1) 
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved to {checkpoint_file}")
        # Add new checkpoint file to the list
        self.checkpoint_files.append(checkpoint_file)
        # Remove the oldest checkpoint if more than 5 are saved
        if len(self.checkpoint_files) > 5:
            oldest_checkpoint = self.checkpoint_files.pop(0)  
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)  # Delete the file
                print(f"Removed old checkpoint: {oldest_checkpoint}")
    def save_best_model(self, model):
        """Save the current model as the best model."""
        torch.save(self.best_model_state, self.save_path)
        print(f"Best model saved to {self.save_path}")