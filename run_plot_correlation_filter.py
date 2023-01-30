# SPDX-License-Identifier: BSD-3-Clause

"""
This script plots the correlation matrix heatmap for the data before and after filtering.
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
    random_state = rng.integers(np.iinfo(np.int32).max)

    #***********************************************************************************************
    # Load the data.
    #***********************************************************************************************
    data = data_factory.make_data()
    load_classes = data_factory.make_load_classes()

    labels = data_factory.make_labels(percent_error=0.4, random_state=random_state)

    start = 0
    timesteps_per_day = 96
    days = 7
    width = days * timesteps_per_day

    data = data[start:(start + width), :]

    #***********************************************************************************************
    # Make the time series' stationary.
    #***********************************************************************************************
    class_cutoffs = {"Residential": 0.06, "Commercial_SM": 0.07, "Commercial_MD": 0.04}

    data_s = transform.filter_butterworth_by_class(
        data=data, class_cutoffs=class_cutoffs, load_classes=load_classes, order=10)

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
    (fig, _) = plot_factory.make_correlation_heatmap(data=data, labels=labels["true"])
    (fig_s, _) = plot_factory.make_correlation_heatmap(data=data_s, labels=labels["true"])

    fig.suptitle("Unfiltered")
    fig_s.suptitle("Filtered")

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
