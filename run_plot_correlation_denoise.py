# SPDX-License-Identifier: BSD-3-Clause

"""
This script plots correlation matrices for the noisy and denoised data.
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
    # We want the randomness to be repeatable.
    rng = np.random.default_rng(seed=1337)

    #***********************************************************************************************
    # Load the data.
    #***********************************************************************************************
    start = 0
    timesteps_per_day = 96
    days = 7
    width = days * timesteps_per_day

    data = data_factory.make_data()[start:(start + width), :]
    load_classes = data_factory.make_load_classes()

    random_state = rng.integers(np.iinfo(np.int32).max)
    labels = data_factory.make_labels(percent_error=0.4, random_state=random_state)

    #***********************************************************************************************
    # Create noisy data.
    #***********************************************************************************************
    data_n = data_factory.make_data_gauss_noise(data=data, percent=0.005, random_state=random_state)

    #data_n = data_factory.make_data_laplace_noise(
    #    data=data, percent=0.005, random_state=random_state
    #)

    #***********************************************************************************************
    # Denoise with SVD.
    #***********************************************************************************************
    (data_nd, values, value_count) = transform.denoise_svd(data=data_n)

    #***********************************************************************************************
    # Make the time series' stationary.
    # NOTE: SVD for denoising doesn't care if the data is stationary or not.
    #***********************************************************************************************
    class_cutoffs = {"Residential": 0.06, "Commercial_SM": 0.07, "Commercial_MD": 0.04}

    data_s = transform.filter_butterworth_by_class(
        data=data, class_cutoffs=class_cutoffs, load_classes=load_classes, order=10
    )

    data_ns = transform.filter_butterworth_by_class(
        data=data_n, class_cutoffs=class_cutoffs, load_classes=load_classes, order=10
    )

    data_nds = transform.filter_butterworth_by_class(
        data=data_nd, class_cutoffs=class_cutoffs, load_classes=load_classes, order=10
    )

    #***********************************************************************************************
    # Calculate Frobenius error.
    #***********************************************************************************************
    error_ns = np.sqrt(np.sum((data_s - data_ns)**2))
    error_nds = np.sqrt(np.sum((data_s - data_nds)**2))

    print('-'*50)
    print(f"Frobenius Error (Noisy): {error_ns}")
    print(f"Frobenius Error (Denoised): {error_nds}")
    print('-'*50)

    #***********************************************************************************************
    # Set AeStHeTiCs.
    #***********************************************************************************************
    #fontsize = 40
    #plt.rc("axes", labelsize=fontsize)
    #plt.rc("axes", titlesize=fontsize)
    #plt.rc("figure", titlesize=fontsize)
    #plt.rc("xtick", labelsize=fontsize)
    #plt.rc("ytick", labelsize=fontsize)

    font = {"family": "monospace", "monospace": ["Times New Roman"]}
    plt.rc("font", **font)

    #***********************************************************************************************
    # Plot correlation matrices.
    #***********************************************************************************************
    (fig_ns, _) = plot_factory.make_correlation_heatmap(data=data_ns, labels=labels["true"])
    (fig_nds, _) = plot_factory.make_correlation_heatmap(data=data_nds, labels=labels["true"])

    fig_ns.suptitle("Noisy")
    fig_nds.suptitle("Denoised")

    #***********************************************************************************************
    # Plot log of singular values.
    #***********************************************************************************************
    (fig_s, axs_s) = plot_factory.make_value_plot(
        values=values[:20], value_count=value_count, size=400
    )

    fig_s.suptitle("Singular Values")

    axs_s.set_yticks(np.arange(start=3, stop=13.5, step=1))

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
