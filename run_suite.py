# SPDX-License-Identifier: MIT

"""
This script runs phase identification across the entire year of data.
"""

import sys

from typing import Any, Dict
from nptyping import NDArray

from rich.progress import track

import numpy as np
import pandas as pd
import sklearn.metrics

import data_factory
import model
import transform

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def _run_identification(
    labels: NDArray[(Any,), int], load: NDArray[(Any, Any), float],
    noise:NDArray[(Any, Any), float], rng: np.random.Generator
    ) -> Dict[str, Any]:
    """
    Run phase identification.

    Parameters
    ----------
    labels : numpy.ndarray of int, (n_load,)
        The phase labels.
    load : numpy.ndarray of float, (n_timestep, n_load)
        The window of load data.
    noise : numpy.ndarray of float, (n_timestep, n_load)
        The window of noisy data.
    rng : numpy.random.Generator
        The random generator to use.

    Returns
    -------
    dict, ['noise_accuracy', 'denoise_accuracy', 'noise_error', 'denoise_error',
    'singular_value_count', 'random_seed']
        The results of the case run.
    """
    (denoise, singular_value_count) = transform.denoise_svd(noise=noise)

    dd_noise = transform.difference(data=noise, order=2)
    dd_denoise = transform.difference(data=denoise, order=2)

    # We want to save the random seed for each window processed incase we need to examine outliers.
    random_seed = rng.integers(np.iinfo(np.int32).max)

    pred_noise = model.predict(labels=labels, load=dd_noise, random_seed=random_seed)
    pred_denoise = model.predict(labels=labels, load=dd_denoise, random_seed=random_seed)

    acc_noise = sklearn.metrics.accuracy_score(y_true=labels, y_pred=pred_noise)
    acc_denoise = sklearn.metrics.accuracy_score(y_true=labels, y_pred=pred_denoise)

    results = {
        "noise_accuracy": acc_noise,
        "denoise_accuracy": acc_denoise,
        "noise_error": np.sqrt(np.sum((noise - load)**2)),
        "denoise_error": np.sqrt(np.sum((denoise - load)**2)),
        "singular_value_count": singular_value_count,
        "random_seed": random_seed
    }

    return results

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals
    """
    The main function.

    The reason for using command line arguments is so this script can be used in an array job on
    high performance computing resources, if available.

    Arguments
    ---------
    days : int
        The size of the window to use in days.
    noise_percent : float
        The percent of noise to inject into the data.

    Examples
    --------
    The following will run phase identification across the entire year of data using a sliding 7-day
    window and an injected noise level of 0.5%:

    python3 run_suite.py 7 0.005
    """
    #***********************************************************************************************
    # Get command line arguemnts.
    #***********************************************************************************************
    days = int(sys.argv[1])
    noise_percent = float(sys.argv[2])

    #***********************************************************************************************
    # Load data.
    #***********************************************************************************************
    # We want the results to be repeatable.
    rng = np.random.default_rng(seed=1337)

    (load, noise) = data_factory.make_data(noise_percent=noise_percent, rng=rng)
    labels = data_factory.make_labels()

    timesteps_per_day = 96
    width = days * timesteps_per_day

    #***********************************************************************************************
    # Calculate the index for each window in the data.
    #***********************************************************************************************
    start_indices = np.arange(
        start=0, stop=load.shape[0] - timesteps_per_day * (days - 1), step=timesteps_per_day
    )

    # Comment out the above statement and uncomment this statement to run the suite on a subset of
    # the data.
    #start_indices = np.arange(
    #    start=0, stop=timesteps_per_day * 30, step=timesteps_per_day
    #)

    #***********************************************************************************************
    # Run the simulation case.
    #***********************************************************************************************
    results_df = pd.DataFrame(data={
        "noise_accuracy": np.zeros(shape=start_indices.size, dtype=float),
        "denoise_accuracy": np.zeros(shape=start_indices.size, dtype=float),
        "noise_error": np.zeros(shape=start_indices.size, dtype=float),
        "denoise_error": np.zeros(shape=start_indices.size, dtype=float),
        "singular_value_count": np.zeros(shape=start_indices.size, dtype=int),
        "random_seed": np.zeros(shape=start_indices.size, dtype=int)
    })

    for (i, start) in track(enumerate(start_indices), "Processing...", total=len(start_indices)):
        win_load = load[start:(start + width), :]
        win_noise = noise[start:(start + width), :]

        results = _run_identification(labels=labels, load=win_load, noise=win_noise, rng=rng)

        for (key, value) in results.items():
            results_df.at[i, key] = value

    #***********************************************************************************************
    # Save results.
    #***********************************************************************************************
    noise_string = str(noise_percent).replace(".", "p")

    results_df.to_csv(f"results/result-{days}-day-{noise_string}-noise.csv")

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
