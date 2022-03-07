# SPDX-License-Identifier: MIT

"""
This script runs phase identification across the entire year of data.
"""

import sys

from rich.progress import track

import numpy as np
import pandas as pd

import data_factory
import simulator

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals
    """
    The main function.

    The reason for using command line arguments is so this script can be used in an array job on
    high performance computing resources, if available.

    Arguments
    ---------
    case_name : str, [
        'case_iqr_svd_svht', 'case_null', 'case_svd_svht'
    ]
        The name of the simulation case to run.
    cpu_count : int
        The number of CPUs cores to use when running the IQR detector. Set to 0 to use all available
        cores.
    days : int
        The size of the window to use in days.
    percent_noise : float
        The percent of noise to inject into the data.
    run_count : int
        The number of times to run the clustering. The clustering with the smallest inertia is used.

    Examples
    --------
    The following will run phase identification across the entire year of data using a sliding 7-day
    window and an injected noise level of 0.5%:

    python3 run_case.py case_null 8 7 0.005 5
    """
    #***********************************************************************************************
    # Get command line arguemnts.
    #***********************************************************************************************
    case_name = str(sys.argv[1])
    cpu_count = int(sys.argv[2])
    days = int(sys.argv[3])
    percent_noise = float(sys.argv[4])
    run_count = int(sys.argv[5])

    cpu_count = cpu_count if cpu_count != 0 else None

    #***********************************************************************************************
    # Load data.
    #***********************************************************************************************
    # We want the results to be repeatable.
    rng = np.random.default_rng(seed=1337)

    data = data_factory.make_data()

    #***********************************************************************************************
    # Load the phase labels and load classes.
    #***********************************************************************************************
    random_state_labels = rng.integers(np.iinfo(np.int32).max)
    labels = data_factory.make_labels(percent_error=0.4, random_state=random_state_labels)

    load_classes = data_factory.make_load_classes()

    #***********************************************************************************************
    # Calculate the index for each window in the data.
    #***********************************************************************************************
    # The choice of which dataset to use for `total_timesteps` doesn't matter. We just want the
    # number of timesteps in a year.
    timesteps_per_day = 96
    width = days * timesteps_per_day

    start_indices = np.arange(
        start=0, stop=data.shape[0] - timesteps_per_day * (days - 1), step=timesteps_per_day
    )

    # Comment out the above statement and uncomment this statement to run the suite on a subset of
    # the data.
    #start_indices = np.arange(
    #    start=0, stop=timesteps_per_day * 20, step=timesteps_per_day
    #)

    #***********************************************************************************************
    # Run the simulation case.
    #***********************************************************************************************
    results_df = pd.DataFrame(data={
        "accuracy_cauchy": np.zeros(shape=len(start_indices), dtype=float),
        "accuracy_gauss": np.zeros(shape=len(start_indices), dtype=float),
        "accuracy_laplace": np.zeros(shape=len(start_indices), dtype=float),
        "accuracy_mixed": np.zeros(shape=len(start_indices), dtype=float),
        "error_denoise_cauchy": np.zeros(shape=len(start_indices), dtype=float),
        "error_denoise_gauss": np.zeros(shape=len(start_indices), dtype=float),
        "error_denoise_laplace": np.zeros(shape=len(start_indices), dtype=float),
        "error_denoise_mixed": np.zeros(shape=len(start_indices), dtype=float),
        "error_noise_cauchy": np.zeros(shape=len(start_indices), dtype=float),
        "error_noise_gauss": np.zeros(shape=len(start_indices), dtype=float),
        "error_noise_laplace": np.zeros(shape=len(start_indices), dtype=float),
        "error_noise_mixed": np.zeros(shape=len(start_indices), dtype=float),
        "random_state": np.zeros(shape=len(start_indices), dtype=int),
        "random_state_labels": np.repeat(random_state_labels, len(start_indices)),
        "value_count_cauchy": np.zeros(shape=len(start_indices), dtype=int),
        "value_count_gauss": np.zeros(shape=len(start_indices), dtype=int),
        "value_count_laplace": np.zeros(shape=len(start_indices), dtype=int),
        "value_count_mixed": np.zeros(shape=len(start_indices), dtype=int),
        "window_start_index": np.zeros(shape=len(start_indices), dtype=int)
    })

    # Noise parameters.
    max_value = 240 * 0.05
    percent_cauchy = 0.0035
    percent_gauss = percent_noise
    percent_laplace = percent_noise

    for (i, start) in track(enumerate(start_indices), "Processing...", total=len(start_indices)):
        random_state = rng.integers(np.iinfo(np.int32).max)
        window = data[start:(start + width), :]

        window_map = {
            "cauchy": data_factory.make_data_cauchy_noise(
                data=window, max_value=max_value, percent=percent_cauchy,
                random_state=random_state
            ),
            "gauss": data_factory.make_data_gauss_noise(
                data=window, percent=percent_gauss, random_state=random_state
            ),
            "laplace": data_factory.make_data_laplace_noise(
                data=window, percent=percent_laplace, random_state=random_state
            ),
            "mixed": data_factory.make_data_mixed_noise(
                data=window, max_value=max_value, percent_cauchy=percent_cauchy,
                percent_gauss=percent_gauss, random_state=random_state
            )
        }

        for (noise_name, window_n) in window_map.items():
            results = simulator.run_case(
                data=window, data_n=window_n, case_name=case_name, cpu_count=cpu_count,
                labels=labels, load_classes=load_classes, random_state=random_state,
                run_count=run_count
            )

            for (metric_name, value) in results.items():
                key = f"{metric_name}_{noise_name}"
                results_df.at[i, key] = value

        results_df.at[i, "random_state"] = random_state
        results_df.at[i, "window_start_index"] = start

    #***********************************************************************************************
    # Save results.
    #***********************************************************************************************
    noise_string = str(percent_noise).replace(".", "p")

    results_df.to_csv(
        f"results/result-{case_name}-{days}-day-{noise_string}-noise-{run_count}-run.csv"
    )

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
