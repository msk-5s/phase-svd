# SPDX-License-Identifier: MIT

"""
This module contains a factory for data.
"""

from typing import Any, Tuple
from nptyping import NDArray

import pyarrow.feather
import numpy as np

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_data(noise_percent: float, rng: np.random.Generator) ->\
Tuple[NDArray[(Any, Any), float], NDArray[(Any, Any), float]]:
    """
    Make a window of load data.

    Parameters
    ----------
    noise_percent : float
        The percentage of noise to add to the data.
    rng : numpy.random.Generator
        The random state generator to use.

    Returns
    -------
    load : numpy.ndarray of float, (n_timestep, n_load)
        The load voltage magnitude data.
    noise : numpy.ndarray of float, (n_timestep, n_load)
        The noisy load voltage magnitude data.
    """
    load = pyarrow.feather.read_feather("data/load_voltage.feather").to_numpy(dtype=float)

    # Inject gaussian white noise into the measurements. With the original value as the mean, set
    # the 3-sigma point as the percentage of noise. This ensures that the probability of a noisy
    # sample being within `noise_percent` of the true value is ~99.7% (68/95/99.7 rule).
    noise = load + rng.normal(loc=0, scale=(load * noise_percent) / 3, size=load.shape)

    return (load, noise)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_labels() -> NDArray[(Any,), int]:
    """
    Make the labels for the loads.

    Returns
    -------
    labels : numpy.ndarray of int, (n_load,)
        The phase labels for each load.
    """
    metadata = pyarrow.feather.read_feather("data/metadata.feather")

    labels = metadata["phase"].to_numpy(dtype=int)

    return labels
