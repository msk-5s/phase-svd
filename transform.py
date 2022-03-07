# SPDX-License-Identifier: MIT

"""
This module contains functions for performing transformations on data.
"""

import multiprocessing as mp
import os

from typing import Any, List, Mapping, Optional, Tuple
from nptyping import NDArray

import numpy as np

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def filter_butterworth(
    data: NDArray[(Any, Any), float], cutoff: float, order: int, filter_type: str = "highpass"
) -> NDArray[(Any, Any), float]:
    """
    Apply a butterworth high pass filter to each column (time series) of `data` to filter out the
    low frequency trend and seasonality.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The time series data to transform.
    cutoff : float
        The cutoff frequency in cycles per sample.
    order : int
        The order of the butterworth filter.
    filter_type : str, ["lowpass", "highpass"], default="highpass"
        The filter type to use.

    Returns
    -------
    numpy.ndarray, (n_timestep, n_series)
        The transformed time series data.
    """
    valid_types = ["lowpass", "highpass"]

    if filter_type not in valid_types:
        ValueError(f"Invalid filter type: {filter_type} - Valid Types: {valid_types}")

    # We add a small number to the frequencies to prevent division by zero.
    frequencies = np.fft.fftfreq(data.shape[0]) + 1e-6

    filter_b = {
        "highpass": (1 / np.sqrt(1 + (cutoff / frequencies)**(2 * order))).astype(complex),
        "lowpass": (1 / np.sqrt(1 + (frequencies / cutoff)**(2 * order))).astype(complex)
    }[filter_type]

    filtered_data = np.zeros(shape=data.shape)

    for (i, series) in enumerate(data.T):
        dft_series = np.fft.fft(series)
        dft_filtered_series = dft_series * filter_b

        filtered_data[:, i] = np.fft.ifft(dft_filtered_series).real

    return filtered_data

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def filter_butterworth_by_class( # pylint: disable=too-many-locals
    data: NDArray[(Any, Any), float], class_cutoffs: Mapping[str, float],
    load_classes: NDArray[(Any,), str], order: int, filter_type: str = "highpass"
) -> NDArray[(Any, Any), float]:
    """
    Apply a butterworth high pass filter to each column (time series) of `data` to filter out the
    low frequency trend and seasonality.

    The cutoff frequencies defined in `cutoffs` will be used for filtering each appropriate load
    class.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The time series data to transform.
    class_cutoffs : dict of (str, float)
        The cutoff frequencies (value) in cycles per sample for each load class (str).
    load_classes : numpy.ndarray of str, (n_load,)
        The class of each load.
    order : int
        The order of the butterworth filter.
    filter_type : str, ["lowpass", "highpass"], default="highpass"
        The filter type to use.

    Returns
    -------
    numpy.ndarray, (n_timestep, n_series)
        The transformed time series data.
    """
    valid_types = ["lowpass", "highpass"]

    if filter_type not in valid_types:
        ValueError(f"Invalid filter type: {filter_type} - Valid Types: {valid_types}")

    #***********************************************************************************************
    # Create a Butterworth highpass filter for each load class.
    #***********************************************************************************************
    # We add a small number to the frequencies to prevent division by zero.
    frequencies = np.fft.fftfreq(data.shape[0]) + 1e-6

    hpfs = {}

    for (load_class, cutoff) in class_cutoffs.items():
        hpfs[load_class] = {
            "highpass": (1 / np.sqrt(1 + (cutoff / frequencies)**(2 * order))).astype(complex),
            "lowpass": (1 / np.sqrt(1 + (frequencies / cutoff)**(2 * order))).astype(complex)
        }[filter_type]

    #***********************************************************************************************
    # Filter the data.
    #***********************************************************************************************
    filtered_data = np.zeros(shape=data.shape)

    for (load_class, hpf) in hpfs.items():
        # Get the indices of loads that are part of the current `load_class`.
        indices = np.where(load_classes == load_class)[0]

        for i in indices:
            series = data[:, i]
            dft_series = np.fft.fft(series)
            dft_filtered_series = dft_series * hpf

            filtered_data[:, i] = np.fft.ifft(dft_filtered_series).real

    return filtered_data

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def filter_impulse_iqr(
    data: NDArray[(Any, Any), float], cpu_count: Optional[int] = None
) -> Tuple[NDArray[(Any, Any), float], List[NDArray[(Any,), int]]]:
    """
    Remove impulse noise from the columns of `data` using a non-linear filter.

    An Interquartile Range (IQR) anomaly detector is used to classify timesteps that contain an
    impulse as an anomaly. The value at each anomalous timestep is then replaced with the mean of
    of the non-anomalous values at that timestep across loads. Note that each time series should be
    stationary (IQR detector performance will degrade with non-stationary data).

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The `data` to filter. The time series in each column should be stationary.
    cpu_count : optional of int, default=None
        The number of CPU cores to use.

    Returns
    -------
    result : numpy.ndarray, (n_timestep, n_series)
        The filtered `data`.
    normal_mask : numpy.ndarray of bool, (n_timestep, n_load)
        The mask of normal (True) and anomalous (False) timesteps.
    """
    #***********************************************************************************************
    # Find the anomalies.
    #***********************************************************************************************
    # Split the data up by the number of CPU's on the system.
    # NOTE: It's important to pay attention to how many CPU cores are reported on the device. For
    # instance, if 8 cores are requested on a 128 core machine with SMT (Simulataneous
    # Multi-Threading), then `os.cpu_count()` will return 256 cores. This can lead to
    # out-of-memory issues which can be resolved by setting `cpu_count` manually.
    cpu_count = cpu_count if cpu_count else os.cpu_count()
    indices_splits = np.array_split(np.arange(data.shape[1]), cpu_count)

    # See `_filter_impulse_iqr` for the required function arguments.
    # Note the comma in the single element tuple. This is required to define a tuple of one element.
    arguments = [(data[:, indices],) for indices in indices_splits]

    # Run the IQR filtering across multiple CPU's. Each process will perform the IQR filtering on a
    # subset of the loads.
    with mp.Pool(cpu_count) as pool:
        # `starmap` will return a list containing the return values of
        # `_filter_impulse_iqr` for each process.
        result_slices = pool.starmap(_filter_impulse_iqr, arguments)

    normal_mask = np.column_stack(result_slices)

    #***********************************************************************************************
    # Replace the anomalous values with the load-wise (column-wise) mean at each respective
    # timestep.
    #***********************************************************************************************
    data_f = data.copy()

    for (i, mask) in enumerate(normal_mask):
        # Only use the mean if there is at least 1 normal measurement at the current timestep.
        if len(np.where(mask)[0]) > 0:
            data_f[i, ~mask] = np.mean(data[i, mask])
        else:
            data_f[i, :] = 0

    return (data_f, normal_mask)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def _filter_impulse_iqr(
    data_slice: NDArray[(Any, Any), float]
) -> NDArray[(Any, Any), float]:
    """
    This function performs IQR filtering on a subset of loads.

    Parameters
    ----------
    data_slice : numpy.ndarray of float, (n_timestep, n_load)
        The subset of load data.

    Returns
    -------
    numpy.ndarray of bool, (n_timestep, n_load)
        The mask of normal (True) and anomalous (False) timesteps.
    """
    normal_mask = np.ones(shape=data_slice.shape, dtype=bool)

    for (load_index, series) in enumerate(data_slice.T):
        (qrt_1, qrt_3) = np.quantile(a=series, q=[0.25, 0.75])
        iqr = qrt_3 - qrt_1

        # 0: Normal | 1: Anomaly
        anomalies = np.array([
            1 if (x > qrt_3 + (iqr * 1.5)) or (x < qrt_1 - (iqr * 1.5)) else 0
        for x in series])

        anomaly_indices = np.where(anomalies == 1)[0]
        normal_mask[anomaly_indices, load_index] = False

    return normal_mask

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def denoise_svd(
    data: NDArray[(Any, Any), float], value_count: Optional[int] = None
) -> Tuple[NDArray[(Any, Any), float], int]:
    """
    Denoise the load data using Singular Value Decomposition (SVD).

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_series)
        The noisy load data to denoise.
    value_count : optional of int, default=None
        The number of singular values to use for reconstruction. If `None` is passed, then the
        Singular Value Hard Threshold (SVHT) is used to determine the number of singular values.

    Returns
    -------
    data_d : numpy.ndarray, (n_timestep, n_series)
        The denoised load data.
    values : numpy.ndarray, (n_series)
        The singular values of `data_d`.
    value_count : int
        The number of singular values kept.
    """
    # pylint: disable=invalid-name
    (u, s, vt) = np.linalg.svd(data, full_matrices=False)

    if value_count is None:
        # Calculate the Singular Value Hard Threshold (SVHT) as presented in "The Optimal Hard
        # Threshold for Singular Values is 4 / sqrt(3)" by Gavish, et al.
        beta = data.shape[0] / data.shape[1]
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        threshold = omega * np.median(s)

        # At high enough noise levels, it is possible for the `value_count` to be 0. In these cases,
        # we use at least 1 value.
        value_count = len(s[s >= threshold])
        value_count = value_count if value_count > 0 else 1

    data_d = u[:, :value_count] @ np.diag(s[:value_count]) @ vt[:value_count, :]

    return (data_d, s, value_count)
