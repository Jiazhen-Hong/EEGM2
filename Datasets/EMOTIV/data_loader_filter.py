from torch.utils.data import DataLoader, TensorDataset
import torch

import numpy as np
from scipy.signal import iirdesign, zpk2sos, sosfiltfilt, cheb2ord

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





def chebyBandpassFilter(data, cutoff, gstop=40, gpass=0.5, fs=128):
    """
    Band-pass filter for data with scipy functions, supporting 1D, 2D, and 3D data.

    Parameters
    ----------
    data : instance of numpy.array | instance of pandas.core.DataFrame
        Data to be filtered. For 3D data (trials, channels, time points), 
        the filter will be applied to each trial and each channel.
    cutoff : array-like of float
        Pass and stop frequencies in order:
            - the first element is the stop limit in the lower bound
            - the second element is the lower bound of the pass-band
            - the third element is the upper bound of the pass-band
            - the fourth element is the stop limit in the upper bound
        For instance, [0.9, 1, 45, 48] will create a band-pass filter between
        1 Hz and 45 Hz.
    gstop : int
        The minimum attenuation in the stopband (dB).
    gpass : int
        The maximum loss in the passband (dB).
    fs : int
        Sampling frequency of the data.

    Returns
    -------
    filteredData : numpy.array
        The filtered data with the same shape as the input.
    """

    # Calculate normalized pass and stop band frequencies
    wp = [cutoff[1] / (fs / 2), cutoff[2] / (fs / 2)]
    ws = [cutoff[0] / (fs / 2), cutoff[3] / (fs / 2)]

    # Design Chebyshev type II filter
    z, p, k = iirdesign(wp=wp, ws=ws, gstop=gstop, gpass=gpass,
                        ftype='cheby2', output='zpk')
    sos = zpk2sos(z, p, k)

    # Handle different dimensions of data
    if data.ndim == 1:
        # 1D data
        filteredData = sosfiltfilt(sos, data)

    elif data.ndim == 2:
        # 2D data (channels, time points)
        filteredData = np.zeros_like(data)
        for electrode in range(data.shape[0]):
            filteredData[electrode] = sosfiltfilt(sos, data[electrode])

    elif data.ndim == 3:
        # 3D data (trials, channels, time points)
        filteredData = np.zeros_like(data)
        for trial in range(data.shape[0]):
            for channel in range(data.shape[1]):
                filteredData[trial, channel] = sosfiltfilt(sos, data[trial, channel])

    else:
        raise ValueError(f"Unsupported data dimensions: {data.ndim}")

    return filteredData