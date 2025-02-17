# #
import numpy as np
from scipy.signal import iirdesign, zpk2sos, sosfiltfilt, cheb2ord

# def chebyBandpassFilter(data, cutoff, gstop=40, gpass=0.5, fs=128):
#     """
#     Design a filter with scipy functions avoiding unstable results (when using
#     ab output and filtfilt(), lfilter()...).
#     Cf. ()[]

#     Parameters
#     ----------
#     data : instance of numpy.array | instance of pandas.core.DataFrame
#         Data to be filtered. Each column will be filtered if data is a
#         dataframe.
#     cutoff : array-like of float
#         Pass and stop frequencies in order:
#             - the first element is the stop limit in the lower bound
#             - the second element is the lower bound of the pass-band
#             - the third element is the upper bound of the pass-band
#             - the fourth element is the stop limit in the upper bound
#         For instance, [0.9, 1, 45, 48] will create a band-pass filter between
#         1 Hz and 45 Hz.
#     gstop : int
#         The minimum attenuation in the stopband (dB).
#     gpass : int
#         The maximum loss in the passband (dB).

#     Returns:

#     zpk :

#     filteredData : instance of numpy.array | instance of pandas.core.DataFrame
#         The filtered data.
#     """

#     wp = [cutoff[1] / (fs / 2), cutoff[2] / (fs / 2)]
#     ws = [cutoff[0] / (fs / 2), cutoff[3] / (fs / 2)]

#     z, p, k = iirdesign(wp=wp, ws=ws, gstop=gstop, gpass=gpass,
#                         ftype='cheby2', output='zpk')
#     zpk = [z, p, k]
#     sos = zpk2sos(z, p, k)

#     order, Wn = cheb2ord(wp=wp, ws=ws, gstop=gstop, gpass=gpass, analog=False)

#     # print('Creating cheby filter of order %d...' % order)

#     if (data.ndim == 2):
#         # print('Data contain multiple columns. Apply filter on each columns.')
#         filteredData = np.zeros(data.shape)
#         for electrode in range(data.shape[1]):
#             # print 'Filtering electrode %s...' % electrode
#             filteredData[:, electrode] = sosfiltfilt(sos, data[:, electrode])
#     else:
#         # Use sosfiltfilt instead of filtfilt fixed the artifacts at the beggining
#         # of the signal
#         filteredData = sosfiltfilt(sos, data)
#     return filteredData


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