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
    # We want and randomness to be repeatable.
    rng = np.random.default_rng(seed=1337)

    #***********************************************************************************************
    # Load the data.
    #***********************************************************************************************
    (load, noise) = data_factory.make_data(noise_percent=0.005, rng=rng)
    labels = data_factory.make_labels()

    days = 7
    timesteps_per_day = 96
    width = days * timesteps_per_day
    start = 0

    load = load[start:(start + width), :]
    noise = noise[start:(start + width), :]

    #***********************************************************************************************
    # Denoise data and calculate Frobenius errors.
    #***********************************************************************************************
    (denoise, value_count) = transform.denoise_svd(noise=noise)

    noise_error = np.sqrt(np.sum((noise - load)**2))
    denoise_error = np.sqrt(np.sum((denoise - load)**2))

    print("-"*50)
    print(f"Singular Value Count: {value_count}")
    print(f"Noise Frobenius Error: {noise_error}")
    print(f"Denoise Frobenius Error: {denoise_error}")
    print("-"*50)

    #***********************************************************************************************
    # Make the time series' stationary.
    #***********************************************************************************************
    dd_load = transform.difference(data=load, order=2)
    dd_noise = transform.difference(data=noise, order=2)
    dd_denoise = transform.difference(data=denoise, order=2)

    #***********************************************************************************************
    # Run phase identification.
    #***********************************************************************************************
    # The `random_seed` in the results can be used instead, to examine a specific snapshot in the
    # year. Be sure to set the correct `start` of the window.
    random_seed = rng.integers(np.iinfo(np.int32).max)

    load_predictions = model.predict(labels=labels, load=dd_load, random_seed=random_seed)
    noise_predictions = model.predict(labels=labels, load=dd_noise, random_seed=random_seed)
    denoise_predictions = model.predict(labels=labels, load=dd_denoise, random_seed=random_seed)

    accuracies = {
        "load": sklearn.metrics.accuracy_score(y_true=labels, y_pred=load_predictions),
        "noise": sklearn.metrics.accuracy_score(y_true=labels, y_pred=noise_predictions),
        "denoise": sklearn.metrics.accuracy_score(y_true=labels, y_pred=denoise_predictions)
    }

    #***********************************************************************************************
    # Print out results.
    #***********************************************************************************************
    print("-"*50)

    for (name, accuracy) in accuracies.items():
        print(f"{name} Accuracy: {accuracy}")

    print("-"*50)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
