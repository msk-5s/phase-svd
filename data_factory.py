# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains a factory for data.
"""

import math

from typing import Any, Dict
from nptyping import NDArray

import pyarrow.feather
import numpy as np
import scipy.stats

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_data() -> NDArray[(Any, Any), float]:
    """
    Make the load data.

    Returns
    -------
    numpy.ndarray of float, (n_timestep, n_load)
        The load voltage magnitude data.
    """
    data = pyarrow.feather.read_feather("data/load_voltage.feather").to_numpy(dtype=float)

    return data

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_data_cauchy_noise(
    data: NDArray[(Any, Any), float], max_value: float, percent: float, random_state: int
) -> NDArray[(Any, Any), float]:
    """
    Make the load data with additive Cauchy noise.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The load voltage magnitude data.
    max_value : float
        The maximum value of any impulse created by the cauchy distribution.
    percent : float
        The probability of observing the `max_value` via Cauchy distribution.
    random_state : int
        The state to use for rng.

    Returns
    -------
    numpy.ndarray of float, (n_timestep, n_load)
        The data with additive Cauchy noise.
    """
    scale = np.abs(max_value / np.tan((np.pi / 2) * (percent - 3)))

    cauchy = scipy.stats.cauchy.rvs(
        loc=0, scale=scale, size=data.shape, random_state=random_state
    )

    # Ensure that the impulses never exceed `max_value`.
    cauchy[cauchy > max_value] = max_value
    cauchy[cauchy < -max_value] = -max_value

    return data + cauchy

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_data_gauss_noise(
    data: NDArray[(Any, Any), float], percent: float, random_state: int
) -> NDArray[(Any, Any), float]:
    """
    Make the load data with additive Gaussian noise.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The load voltage magnitude data.
    percent : float
        The percentage of Gaussian noise to add.
    random_state : int
        The state to use for rng.

    Returns
    -------
    numpy.ndarray of float, (n_timestep, n_load)
        The data with additive Gaussian noise.
    """
    # ~99.7% probability of noisy value being within `percent_gauss` of the true value.
    gauss = scipy.stats.norm.rvs(
        loc=0, scale=(data * percent) / 3, size=data.shape, random_state=random_state
    )

    return data + gauss

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_data_laplace_noise(
    data: NDArray[(Any, Any), float], percent: float, random_state: int
) -> NDArray[(Any, Any), float]:
    """
    Make the load data with additive Laplacian noise.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The load voltage magnitude data.
    percent : float
        The percentage of Laplacian noise to add.
    random_state : int
        The state to use for rng.

    Returns
    -------
    numpy.ndarray of float, (n_timestep, n_load)
        The data with additive Laplacian noise.
    """
    laplace = scipy.stats.laplace.rvs(
        loc=0, scale=(data * percent) / 3, size=data.shape, random_state=random_state
    )

    return data + laplace

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_data_mixed_noise(
    data: NDArray[(Any, Any), float], max_value: float, percent_cauchy: float, percent_gauss: float,
    random_state: int
) -> NDArray[(Any, Any), float]:
    """
    Make the load data with additive mixed Cauchy/Gaussian noise.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The load voltage magnitude data.
    max_value : float
        The maximum value of any impulse created by the cauchy distribution.
    percent_cauchy : float
        The probability of observing the `max_value` via Cauchy distribution.
    percent_gauss : float
        The percentage of Gaussian noise to add.
    random_state : int
        The state to use for rng.

    Returns
    -------
    numpy.ndarray of float, (n_timestep, n_load)
        The data with additive Laplacian noise.
    """
    scale = np.abs(max_value / np.tan((np.pi / 2) * (percent_cauchy - 3)))

    cauchy = scipy.stats.cauchy.rvs(
        loc=0, scale=scale, size=data.shape, random_state=random_state
    )

    # Ensure that the impulses never exceed `max_value`.
    cauchy[cauchy > max_value] = max_value
    cauchy[cauchy < -max_value] = -max_value

    # ~99.7% probability of noisy value being within `percent_gauss` of the true value.
    gauss = scipy.stats.norm.rvs(
        loc=0, scale=(data * percent_gauss) / 3, size=data.shape, random_state=random_state
    )

    return data + cauchy + gauss

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_labels(percent_error: float, random_state: int) -> Dict[str, NDArray[(Any,), int]]:
    """
    Make the true and erroneous phase labels.

    Parameters
    ----------
    percent_error : float
        The percentage of phase labels to make incorrect.
    random_state: int
        The random state to use for rng.

    Returns
    -------
    dict of [str, NDArray[(Any,), int]]
        error : numpy.ndarray of int, (n_load,)
            The erroneous phase labels.
        true : numpy.ndarray of int, (n_load,)
            The true phase labels.
    """
    labels_true = pyarrow.feather.read_feather("data/metadata.feather")["phase"].to_numpy(dtype=int)

    rng = np.random.default_rng(random_state)

    label_count = len(labels_true)
    error_count = int(label_count * percent_error)
    indices = rng.permutation(label_count)[:error_count]

    unique_count = len(np.unique(labels_true))
    labels_error = labels_true.copy()

    # Increment the original label by 1 and wrap around when appropriate.
    labels_error[indices] = (labels_error[indices] + 1) % unique_count

    labels = {
        "error": labels_error,
        "true": labels_true
    }

    return labels

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_load_classes() -> NDArray[(Any,), str]:
    """
    Make an array containing the class of each load.

    Returns
    -------
    numpy.ndarray of str, (n_load,)
        The load classes.
    """
    metadata_df = pyarrow.feather.read_feather("data/metadata.feather")

    load_classes = metadata_df["loadshape"].to_numpy(dtype=str)

    return load_classes

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_series_windows(index: int, stride: int, width: int) -> NDArray[(Any, Any), float]:
    """
    Make a matrix of windows of the voltage magnitude measurements of the load at `index` in the
    dataset.

    This function is only used as a convenience function for plotting spectrograms.

    Parameters
    ----------
    index : int
        The index of the load's time series to use.
    stride : int
        The stride of the windows in timesteps.
    width : int
        The width of the window in timesteps.

    Returns
    -------
    numpy.ndarray of float, (width, n_window)
        The time series as a matrix of stride-lagged windows.
    """
    data = pyarrow.feather.read_feather("data/load_voltage.feather").to_numpy(dtype=float)
    series = data[:, index]

    window_count = math.ceil((len(series) - width) / stride)
    start_indices = np.array([stride * i for i in range(window_count)])

    windows = np.zeros(shape=(width, window_count))

    for (i, start) in enumerate(start_indices):
        windows[:, i] = series[start:start + width]

    return windows
