# SPDX-License-Identifier: MIT

"""
This script plots the time series, acf, and correlation matrices for different time difference
orders.
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
    (load, _) = data_factory.make_data(noise_percent=0.005, rng=rng)
    labels = data_factory.make_labels()

    start = 0
    width = 96 * 7

    load = load[start:(start + width), :]

    #***********************************************************************************************
    # Make the time series' stationary.
    #***********************************************************************************************
    d_load = transform.difference(data=load, order=1)
    dd_load = transform.difference(data=load, order=2)

    #***********************************************************************************************
    # Set font sizes.
    #***********************************************************************************************
    #fontsize = 40
    #plt.rc("axes", labelsize=fontsize)
    #plt.rc("axes", titlesize=fontsize)
    #plt.rc("figure", titlesize=fontsize)
    #plt.rc("xtick", labelsize=fontsize)
    #plt.rc("ytick", labelsize=fontsize)

    #***********************************************************************************************
    # Plot correlation matrices.
    #***********************************************************************************************
    # Index of the load to plot.
    index = 1337

    (fig_load, _) = plot_factory.make_voltage_series_plot(series=load[:, index])
    (fig_d_load, _) = plot_factory.make_voltage_series_plot(series=d_load[:, index])
    (fig_dd_load, _) = plot_factory.make_voltage_series_plot(series=dd_load[:, index])

    fig_load.suptitle("Load")
    fig_d_load.suptitle("First Order Difference")
    fig_dd_load.suptitle("Second Order Difference")

    (fig_acf_load, _) = plot_factory.make_acf_plot(series=load[:, index], lags=96)
    (fig_acf_d_load, _) = plot_factory.make_acf_plot(series=d_load[:, index], lags=96)
    (fig_acf_dd_load, _) = plot_factory.make_acf_plot(series=dd_load[:, index], lags=96)

    fig_acf_load.suptitle("Load")
    fig_acf_d_load.suptitle("First Order Difference")
    fig_acf_dd_load.suptitle("Second Order Difference")

    (fig_corr_load, _) = plot_factory.make_correlation_heatmap(load=load, labels=labels)
    (fig_corr_d_load, _) = plot_factory.make_correlation_heatmap(load=d_load, labels=labels)
    (fig_corr_dd_load, _) = plot_factory.make_correlation_heatmap(load=dd_load, labels=labels)

    fig_corr_load.suptitle("Load")
    fig_corr_d_load.suptitle("First Order Difference")
    fig_corr_dd_load.suptitle("Second Order Difference")

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
