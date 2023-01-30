# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains different cases for simulation.
"""

from typing import Any, Dict, Mapping, Optional, Tuple
from nptyping import NDArray

import numpy as np
import sklearn.metrics

import model
import transform

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def run_case( # pylint: disable=too-many-arguments, too-many-locals
    data: NDArray[(Any, Any), float], data_n: NDArray[(Any, Any), float], case_name: str,
    labels: Mapping[str, NDArray[(Any,), float]], load_classes: NDArray[(Any,), str],
    random_state: int, run_count: int, cpu_count: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run a simulation case.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The noiseless data.
    data_n : numpy.ndarray of float, (n_timestep, n_load)
        The noisy data.
    case_name : str, [
        'case_iqr_svd_svht', 'case_null', 'case_svd_svht'
    ]
        The name of the case to run.
    labels : dict of [str, NDArray[(Any,), int]]
        error : numpy.ndarray of int, (n_load,)
            The erroneous phase labels.
        true : numpy.ndarray of int, (n_load,)
            The true phase labels.
    load_classes: numpy.ndarray of str, (n_load,)
        The class of each load.
    random_state : int
        The random seed to use for rng.
    run_count : int
        The number of times to run the clustering. The clustering with the smallest inertia is used.
    cpu_count: optional int, default=None
        The number of CPU cores to use. `None` will use all available cores.

    Returns
    -------
    dict of [str, any]
        accuracy : float
            The phase identification accuracy.
        error_denoise : float
            The Frobenius error for the denoised data.
        error_noise : float
            The Frobenius error for the noisy data.
        value_count : int
            The number of singular values used for denoising.
    """
    #***********************************************************************************************
    # Make the data stationary.
    #***********************************************************************************************
    # The Butterworth filter cutoff frequencies to use for each load class.
    class_cutoffs = {"Residential": 0.06, "Commercial_SM": 0.07, "Commercial_MD": 0.04}

    data_s = transform.filter_butterworth_by_class(
        data=data, class_cutoffs=class_cutoffs, load_classes=load_classes, order=10
    )

    data_ns = transform.filter_butterworth_by_class(
        data=data_n, class_cutoffs=class_cutoffs, load_classes=load_classes, order=10
    )

    #***********************************************************************************************
    # Run the case.
    #***********************************************************************************************
    run = {
        "case_iqr_svd_svht": _run_case_iqr_svd_svht,
        "case_null": lambda data, cpu_count: (data, 0),
        "case_svd_svht": _run_case_svd_svht,
    }[case_name]

    (data_nsd, value_count) = run(data=data_ns, cpu_count=cpu_count)

    predictions = model.predict(
        labels=labels["error"], data=data_nsd, random_state=random_state, run_count=run_count
    )

    #***********************************************************************************************
    # Aggregrate the results.
    #***********************************************************************************************
    results = {
        "accuracy": sklearn.metrics.accuracy_score(y_true=labels["true"], y_pred=predictions),
        "error_denoise": np.sqrt(np.sum((data_s - data_nsd)**2)),
        "error_noise": np.sqrt(np.sum((data_s - data_ns)**2)),
        "value_count": value_count
    }

    return results

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def _run_case_iqr_svd_svht(
    data: NDArray[(Any, Any), float], cpu_count: Optional[int] = None
) -> Tuple[NDArray[(Any, Any), float], int]:
    """
    This case performs IQR filtering on the stationary data before denoising with SVD and SVHT.

    Parameters
    ----------
    data : numpy.ndarray of float, (n_timestep, n_load)
        The data to use.
    cpu_count: optional int, default=None
        The number of CPU cores to use. `None` will use all available cores.

    Returns
    -------
    result : numpy.ndarray of float, (n_timestep, n_load)
        The denoised data.
    value_count : int
        The number of singular values used for matrix reconstruction.
    """
    (data_f, _) = transform.filter_impulse_iqr(data=data, cpu_count=cpu_count)
    (result, _, value_count) = transform.denoise_svd(data=data_f)

    return (result, value_count)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def _run_case_svd_svht( # pylint: disable=unused-argument
    data: NDArray[(Any, Any), float], cpu_count: Optional[int] = None
) -> Tuple[NDArray[(Any, Any), float], int]:
    """
    This case performs denoising with SVD using the Singular Value Hard Threshold (SVHT).

    Parameters
    ----------
    data_n : numpy.ndarray of float, (n_timestep, n_load)
        The data to use.
    cpu_count: optional int, default=None
        The number of CPU cores to use. `None` will use all available cores.

    Returns
    -------
    result : numpy.ndarray of float, (n_timestep, n_load)
        The denoised data.
    value_count : int
        The number of singular values used for matrix reconstruction.
    """
    (result, _, value_count) = transform.denoise_svd(data=data)

    return (result, value_count)
