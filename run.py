# SPDX-License-Identifier: MIT

"""
This script runs phase identification for a single snapshot in time.
"""

import numpy as np
import sklearn.metrics

import model
import transform

import data_factory

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals, too-many-statements
    """
    The main function.
    """
    # We want the randomness to be repeatable.
    rng = np.random.default_rng(seed=1337)
    random_state = rng.integers(np.iinfo(np.int32).max)

    #***********************************************************************************************
    # Load the data.
    #***********************************************************************************************
    labels = data_factory.make_labels(percent_error=0.1, random_state=random_state)
    load_classes = data_factory.make_load_classes()

    days = 7
    timesteps_per_day = 96
    width = days * timesteps_per_day
    start = 0

    data = data_factory.make_data()[start:(start + width), :]

    max_value = 240 * 0.05
    percent_cauchy = 0.0035
    percent_gauss = 0.005
    percent_laplace = 0.005

    data = {
        "cauchy": data_factory.make_data_cauchy_noise(
            data=data, max_value=max_value, percent=percent_cauchy, random_state=random_state
        ),
        "gauss": data_factory.make_data_gauss_noise(
            data=data, percent=percent_gauss, random_state=random_state
        ),
        "laplace": data_factory.make_data_laplace_noise(
            data=data, percent=percent_laplace, random_state=random_state
        ),
        "mixed": data_factory.make_data_mixed_noise(
            data=data, max_value=max_value, percent_cauchy=percent_cauchy,
            percent_gauss=percent_gauss, random_state=random_state
        ),
        "none": data
    }

    #***********************************************************************************************
    # Make the time series' stationary.
    #***********************************************************************************************
    # The Butterworth filter cutoff frequencies to use for each load class.
    class_cutoffs = {"Residential": 0.06, "Commercial_SM": 0.07, "Commercial_MD": 0.04}

    data_s = {}

    for (key, value) in data.items():
        data_s[key] = transform.filter_butterworth_by_class(
            data=value, class_cutoffs=class_cutoffs, load_classes=load_classes, order=10
        )

    #***********************************************************************************************
    # Denoise data and calculate Frobenius errors.
    #***********************************************************************************************
    data_sd = {}
    value_counts = {}

    for (key, value) in data_s.items():
        (value_f, _) = transform.filter_impulse_iqr(data=value)
        (data_sd[key], _, value_counts[key]) = transform.denoise_svd(data=value_f)

    errors = {}
    errors_d = {}

    for key in data_s.keys(): # pylint: disable=consider-using-dict-items, consider-iterating-dictionary
        errors[key] = np.sqrt(np.sum((data_s["none"] - data_s[key])**2))
        errors_d[key] = np.sqrt(np.sum((data_s["none"] - data_sd[key])**2))

    #***********************************************************************************************
    # Run phase identification.
    #***********************************************************************************************
    # The `random_state` in the results can be used instead, to examine a specific snapshot in the
    # year. Be sure to set the correct `start` of the window.
    random_state = rng.integers(np.iinfo(np.int32).max)
    run_count = 1

    accuracies = {}
    accuracies_d = {}

    for (key, value_s, value_sd) in zip(data_s.keys(), data_s.values(), data_sd.values()):
        predictions = model.predict(
            labels=labels["error"], data=value_s, random_state=random_state, run_count=run_count
        )

        predictions_d = model.predict(
            labels=labels["error"], data=value_sd, random_state=random_state, run_count=run_count
        )

        accuracies[key] = sklearn.metrics.accuracy_score(
            y_true=labels["true"], y_pred=predictions
        )

        accuracies_d[key] = sklearn.metrics.accuracy_score(
            y_true=labels["true"], y_pred=predictions_d
        )

    #***********************************************************************************************
    # Print out results.
    #***********************************************************************************************
    print("*"*50)

    for key in data_s.keys(): # pylint: disable=consider-iterating-dictionary
        print("."*10)
        print(f"{key} - Values: {value_counts[key]}")
        print(f"{key} - Accuracy: {accuracies[key]}")
        print(f"{key} - Frobenius Error: {errors[key]}")
        print(f"{key} - Accuracy (Denoised): {accuracies_d[key]}")
        print(f"{key} - Frobenius Error (Denoised): {errors_d[key]}")

    print("*"*50)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
