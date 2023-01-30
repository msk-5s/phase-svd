# SPDX-License-Identifier: BSD-3-Clause

"""
This script plots the a time series with the IQR filter applied to it.
"""

import matplotlib.pyplot as plt
import numpy as np

import data_factory
import plot_factory
import transform

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals,too-many-statements
    """
    The main function.
    """
    # We want the randomness to be repeatable.
    rng = np.random.default_rng(seed=1337)
    random_state = rng.integers(np.iinfo(np.int32).max)

    #***********************************************************************************************
    # Load the data.
    #***********************************************************************************************
    start = 0
    timesteps_per_day = 96
    days = 7
    width = days * timesteps_per_day

    data = data_factory.make_data()[start:(start + width), :]
    load_classes = data_factory.make_load_classes()

    cauchy = data_factory.make_data_cauchy_noise(
        data=data, max_value=240 * 0.05, percent=0.0035, random_state=random_state
    )

    mixed = data_factory.make_data_mixed_noise(
        data=data, max_value=240 * 0.05, percent_cauchy=0.0035, percent_gauss=0.005,
        random_state=random_state
    )

    #***********************************************************************************************
    # Make the time series' stationary.
    #***********************************************************************************************
    class_cutoffs = {"Residential": 0.06, "Commercial_SM": 0.07, "Commercial_MD": 0.04}

    cauchy_s = transform.filter_butterworth_by_class(
        data=cauchy, class_cutoffs=class_cutoffs, load_classes=load_classes, order=10,
        filter_type="highpass"
    )

    mixed_s = transform.filter_butterworth_by_class(
        data=mixed, class_cutoffs=class_cutoffs, load_classes=load_classes, order=10,
        filter_type="highpass"
    )

    #***********************************************************************************************
    # Extract the lower-frequency trend and seasonal components so they can be added back later.
    #***********************************************************************************************
    cauchy_l = transform.filter_butterworth_by_class(
        data=cauchy, class_cutoffs=class_cutoffs, load_classes=load_classes, order=10,
        filter_type="lowpass"
    )

    mixed_l = transform.filter_butterworth_by_class(
        data=mixed, class_cutoffs=class_cutoffs, load_classes=load_classes, order=10,
        filter_type="lowpass"
    )

    #***********************************************************************************************
    # Apply IQR filtering.
    #***********************************************************************************************
    (cauchy_sf, cauchy_mask) = transform.filter_impulse_iqr(data=cauchy_s)
    (mixed_sf, mixed_mask) = transform.filter_impulse_iqr(data=mixed_s)

    #***********************************************************************************************
    # Set AeStHeTiCs.
    #***********************************************************************************************
    #fontsize = 60
    #legend_fontsize = 30
    #plt.rc("axes", labelsize=fontsize)
    #plt.rc("axes", titlesize=fontsize)
    #plt.rc("legend", title_fontsize=legend_fontsize)
    #plt.rc("legend", fontsize=legend_fontsize)
    #plt.rc("figure", titlesize=fontsize)
    #plt.rc("xtick", labelsize=fontsize)
    #plt.rc("ytick", labelsize=fontsize)

    font = {"family": "monospace", "monospace": ["Times New Roman"]}
    plt.rc("font", **font)

    #***********************************************************************************************
    # Get IQR thresholds.
    #***********************************************************************************************
    index = 42

    series_cauchy_s = cauchy_s[:, index]
    series_mixed_s = mixed_s[:, index]

    (qrt_1_c, qrt_3_c) = np.quantile(a=series_cauchy_s, q=[0.25, 0.75])
    (qrt_1_m, qrt_3_m) = np.quantile(a=series_mixed_s, q=[0.25, 0.75])

    iqr_c = qrt_3_c - qrt_1_c
    iqr_m = qrt_3_m - qrt_1_m

    thresholds_cauchy = (qrt_3_c + iqr_c * 1.5, qrt_1_c - iqr_c * 1.5)
    thresholds_mixed = (qrt_3_m + iqr_m * 1.5, qrt_1_m - iqr_m * 1.5)

    #***********************************************************************************************
    # Reconstruct the series' by adding the low-frequency components back.
    #***********************************************************************************************
    series_cauchy_l = cauchy_l[:, index]
    series_mixed_l = mixed_l[:, index]

    series_cauchy_sf = cauchy_sf[:, index]
    series_mixed_sf = mixed_sf[:, index]

    series_cauchy_rf = series_cauchy_sf + series_cauchy_l
    series_mixed_rf = series_mixed_sf + series_mixed_l

    #***********************************************************************************************
    # Plot noise/filtered series.
    #***********************************************************************************************
    (fig_cauchy, _) = plot_factory.make_voltage_series_iqr_plot(
        series_n=series_cauchy_s, series_nf=series_cauchy_sf, thresholds=thresholds_cauchy
    )

    (fig_mixed, _) = plot_factory.make_voltage_series_iqr_plot(
        series_n=series_mixed_s, series_nf=series_mixed_sf, thresholds=thresholds_mixed
    )

    (fig_cauchy_r, _) = plot_factory.make_voltage_series_iqr_plot(
        series_n=cauchy[:, index], series_nf=series_cauchy_rf
    )

    (fig_mixed_r, _) = plot_factory.make_voltage_series_iqr_plot(
        series_n=mixed[:, index], series_nf=series_mixed_rf
    )

    fig_cauchy.suptitle("Cauchy")
    fig_mixed.suptitle("Mixed")
    fig_cauchy_r.suptitle("Cauchy")
    fig_mixed_r.suptitle("Mixed")

    #***********************************************************************************************
    # Plot the normal masks to show normal/anomalous timesteps.
    #***********************************************************************************************
    (fig_cauchy_mask, axs_cauchy_mask) = plt.subplots()
    fig_cauchy_mask.tight_layout()
    axs_cauchy_mask.imshow(cauchy_mask.T, aspect="auto")

    (fig_mixed_mask, axs_mixed_mask) = plt.subplots()
    fig_mixed_mask.tight_layout()
    axs_mixed_mask.imshow(mixed_mask.T, aspect="auto")

    fig_cauchy_mask.suptitle("Cauchy")
    fig_mixed_mask.suptitle("Mixed")

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
