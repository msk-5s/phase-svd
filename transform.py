# SPDX-License-Identifier: MIT

"""
This module contains functions for performing transformations on data.
"""

from typing import Any, Tuple
from nptyping import NDArray

import numpy as np

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def denoise_svd(noise: NDArray[(Any, Any), float]) -> Tuple[NDArray[(Any, Any), float], int]:
    """
    Denoise the load data using Singular Value Decomposition (SVD).

    Parameters
    ----------
    noise : numpy.ndarray, (n_timestep, n_load)
        The noisy load data to denoise.

    Returns
    -------
    denoise : numpy.ndarray, (n_timestep, n_load)
        The denoised load data.
    value_count : int
        The number of singular values kept.
    """
    # pylint: disable=invalid-name
    (u, s, vt) = np.linalg.svd(noise, full_matrices=False)

    # Calculate the Singular Value Hard Threshold (SVHT) as presented in "The Optimal Hard
    # Threshold for Singular Values is 4 / sqrt(3)" by Gavish, et al.
    beta = noise.shape[0] / noise.shape[1]
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    threshold = omega * np.median(s)
    value_count = len(s[s >= threshold])

    denoise = u[:, :value_count] @ np.diag(s[:value_count]) @ vt[:value_count, :]

    return (denoise, value_count)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def difference(data: NDArray[(Any, Any), float], order: int) -> NDArray[(Any, Any), float]:
    """
    Apply a time difference transformation.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_source)
        The time series data to transform.
    order : int
        The order of the time difference to apply.

    Returns
    -------
    numpy.ndarray, (n_timestep - order, n_source)
        The transformed time series data.
    """
    d_data = data

    for _ in range(order):
        d_data = d_data[1:, :] - d_data[:-1, :]

    return d_data
