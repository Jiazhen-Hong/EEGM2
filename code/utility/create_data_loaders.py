from torch.utils.data import DataLoader, TensorDataset
import torch

@staticmethod
def create_data_loaders(train_data, train_label, val_data=None, val_label=None, test_data=None, test_label=None, batch_size=128):
    """
    Create train, validation, and test DataLoaders from pre-split dataset.

    Args:
        train_data: numpy array of training data (samples, channels, time points).
        train_label: numpy array of training labels (samples,).
        val_data: numpy array of validation data (samples, channels, time points), default=None.
        val_label: numpy array of validation labels (samples,), default=None.
        test_data: numpy array of test data (samples, channels, time points), default=None.
        test_label: numpy array of test labels (samples,), default=None.
        batch_size: int, batch size for DataLoader.

    Returns:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data (or None if val_data is None).
        test_loader: DataLoader for test data (or None if test_data is None).
    """
    # Convert training data to PyTorch tensors
    tensor_train_data = torch.tensor(train_data, dtype=torch.float32)
    tensor_train_label = torch.tensor(train_label, dtype=torch.long)
    train_dataset = TensorDataset(tensor_train_data, tensor_train_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optional: Validation data
    val_loader = None
    if val_data is not None and val_label is not None:
        tensor_val_data = torch.tensor(val_data, dtype=torch.float32)
        tensor_val_label = torch.tensor(val_label, dtype=torch.long)
        val_dataset = TensorDataset(tensor_val_data, tensor_val_label)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optional: Test data
    test_loader = None
    if test_data is not None and test_label is not None:
        tensor_test_data = torch.tensor(test_data, dtype=torch.float32)
        tensor_test_label = torch.tensor(test_label, dtype=torch.long)
        test_dataset = TensorDataset(tensor_test_data, tensor_test_label)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader