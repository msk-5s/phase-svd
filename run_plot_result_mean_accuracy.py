# SPDX-License-Identifier: BSD-3-Clause

"""
This script plots the mean accuracy of the results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals, too-many-statements
    """
    The main function.
    """
    noise_percent = "0p005"
    run_count = 10

    noise_names = ["cauchy", "gauss", "laplace", "mixed"]

    #***********************************************************************************************
    # Load the results.
    #***********************************************************************************************
    results_null = {"cauchy": [], "gauss": [], "laplace": [], "mixed": []}
    results_svd_svht = {"cauchy": [], "gauss": [], "laplace": [], "mixed": []}
    results_iqr_svd_svht = {"cauchy": [], "gauss": [], "laplace": [], "mixed": []}

    for day in range(1, 8):
        frame_null = pd.read_csv(
            f"results/result-case_null-{day}-day-{noise_percent}-noise-{run_count}-run.csv",
            index_col=0
        )

        frame_svd_svht = pd.read_csv(
            f"results/result-case_svd_svht-{day}-day-{noise_percent}-noise-{run_count}-run.csv",
            index_col=0
        )

        frame_iqr_svd_svht = pd.read_csv(
            f"results/result-case_iqr_svd_svht-{day}-day-{noise_percent}-noise-{run_count}-run.csv",
            index_col=0
        )

        for noise_name in noise_names:
            mean_null = frame_null[f"accuracy_{noise_name}"].mean()
            mean_svd_svht = frame_svd_svht[f"accuracy_{noise_name}"].mean()
            mean_iqr_svd_svht = frame_iqr_svd_svht[f"accuracy_{noise_name}"].mean()

            results_null[f"{noise_name}"].append(mean_null)
            results_svd_svht[f"{noise_name}"].append(mean_svd_svht)
            results_iqr_svd_svht[f"{noise_name}"].append(mean_iqr_svd_svht)

    #***********************************************************************************************
    # Aggregate results for plotting.
    #***********************************************************************************************
    noisy = {key: results_null[key] for key in noise_names}

    denoised = {
        "cauchy": results_iqr_svd_svht["cauchy"], "gauss": results_svd_svht["gauss"],
        "laplace": results_svd_svht["laplace"], "mixed": results_iqr_svd_svht["mixed"]
    }

    colors = dict(zip(noise_names, ["red", "green", "blue", "orange"]))
    markers = dict(zip(noise_names, ["o", "v", "s", "D"]))

    #***********************************************************************************************
    # Set AeStHeTiCs.
    #***********************************************************************************************
    sns.set_theme(style="whitegrid")

    #fontsize = 40
    #legend_fontsize = 20
    #plt.rc("axes", labelsize=fontsize)
    #plt.rc("legend", title_fontsize=legend_fontsize)
    #plt.rc("legend", fontsize=legend_fontsize)
    #plt.rc("xtick", labelsize=fontsize)
    #plt.rc("ytick", labelsize=fontsize)

    font = {"family": "monospace", "monospace": ["Times New Roman"]}
    plt.rc("font", **font)

    #***********************************************************************************************
    # Plot the results.
    #***********************************************************************************************
    (fig, axs) = plt.subplots()
    fig.tight_layout()

    days = range(1, 8)
    linewidth = 3
    marker_size = 100

    for noise_name in noise_names:
        axs.plot(
            days, noisy[noise_name], color=colors[noise_name], linestyle="dashed",
            linewidth=linewidth, label=f"{noise_name.capitalize()}: Noisy"
        )

        axs.plot(
            days, denoised[noise_name], color=colors[noise_name], linewidth=linewidth,
            label=f"{noise_name.capitalize()}: Denoised"
        )

        axs.scatter(
            days, noisy[noise_name], c=colors[noise_name], marker=markers[noise_name], s=marker_size
        )

        axs.scatter(
            days, denoised[noise_name], c=colors[noise_name], marker=markers[noise_name],
            s=marker_size
        )

    # Change linewidth of the lines shown in the legend.
    _ = [line.set_linewidth(linewidth) for line in axs.legend().get_lines()]

    axs.set_yticks([round(x, 2) for x in np.arange(start=0.35, stop=1.05, step=0.05)])

    axs.set_xlabel("Window Width (Days)")
    axs.set_ylabel("Average Accuracy")
    axs.legend(loc="center right")

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
