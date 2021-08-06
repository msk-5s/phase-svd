# SPDX-License-Identifier: MIT

"""
This script plots correlation matrices for the noiseless, noisy, and denoised data.
"""

import matplotlib.pyplot as plt
import numpy as np

import data_factory
import plot_factory
import transform

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals
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

    start = 0
    width = 96 * 7

    load = load[start:(start + width), :]
    noise = noise[start:(start + width), :]

    #***********************************************************************************************
    # Denoise data and calculate Frobenius errors.
    #***********************************************************************************************
    (u, s, vt) = np.linalg.svd(noise, full_matrices=False) # pylint: disable=invalid-name

    # Calculate the Singular Value Hard Threshold (SVHT) as presented in "The Optimal Hard
    # Threshold for Singular Values is 4 / sqrt(3)" by Gavish, et al.
    beta = noise.shape[0] / noise.shape[1]
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    threshold = omega * np.median(s)
    value_count = len(s[s >= threshold])

    denoise = u[:, :value_count] @ np.diag(s[:value_count]) @ vt[:value_count, :]

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
    # Set font sizes.
    #***********************************************************************************************
    #fontsize = 60
    #plt.rc("axes", labelsize=fontsize)
    #plt.rc("axes", titlesize=fontsize)
    #plt.rc("figure", titlesize=fontsize)
    #plt.rc("xtick", labelsize=fontsize)
    #plt.rc("ytick", labelsize=fontsize)

    #***********************************************************************************************
    # Plot correlation matrices.
    #***********************************************************************************************
    (fig_dd_load, _) = plot_factory.make_correlation_heatmap(load=dd_load, labels=labels)
    (fig_dd_noise, _) = plot_factory.make_correlation_heatmap(load=dd_noise, labels=labels)
    (fig_dd_denoise, _) = plot_factory.make_correlation_heatmap(load=dd_denoise, labels=labels)

    fig_dd_load.suptitle("Load")
    fig_dd_noise.suptitle("Noise")
    fig_dd_denoise.suptitle("Denoise")

    #***********************************************************************************************
    # Plot log of singular values.
    #***********************************************************************************************
    (fig_s, _) = plot_factory.make_value_plot(
        singular_values=s[:30], value_count=value_count, size=400
    )

    fig_s.suptitle("Singular Values")

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
