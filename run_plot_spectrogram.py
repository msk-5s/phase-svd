# SPDX-License-Identifier: BSD-3-Clause

"""
This script plots the spectrograms of a load's voltage magnitude measurements and its filtered
voltage over windows of measurements.
"""

from collections import namedtuple

import matplotlib.pyplot as plt

import data_factory
import plot_factory
import transform

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals
    """
    The main function.
    """
    Parameters = namedtuple(typename="LoadParameters", field_names=["index", "cutoff"])

    load = {
        "Residential": Parameters(index=0, cutoff=0.06),
        "Commercial_SM": Parameters(index=83, cutoff=0.07),
        "Commercial_MD": Parameters(index=259, cutoff=0.04)
    }["Residential"]

    #***********************************************************************************************
    # Load the data.
    #***********************************************************************************************
    days = 7
    timesteps_per_day = 96
    width = timesteps_per_day * days

    windows = data_factory.make_series_windows(
        index=load.index, stride=timesteps_per_day, width=width
    )

    #***********************************************************************************************
    # Filter the windows.
    #***********************************************************************************************
    order = 10

    windows_s = transform.filter_butterworth(data=windows, cutoff=load.cutoff, order=order)

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
    # Plot the virtual measurement spectrograms.
    #***********************************************************************************************
    (fig, _) = plot_factory.make_spectrogram_plot(windows=windows, cutoff=load.cutoff)
    (fig_s, _) = plot_factory.make_spectrogram_plot(windows=windows_s, cutoff=load.cutoff)

    fig.suptitle("Unfiltered Series")
    fig_s.suptitle("Filtered Series")

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
