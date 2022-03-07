# SPDX-License-Identifier: MIT

"""
This script plots the accuracy results for different number of runs.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals, too-many-statements
    """
    The main function.
    """
    case_name = "case_iqr_svd_svht"
    noise_name = "cauchy"
    noise_percent = "0p005"
    run_count_left = 1
    run_count_right = 10

    #***********************************************************************************************
    # Load the results and place them in long form.
    #***********************************************************************************************
    results = pd.DataFrame(columns=["accuracy", "day", "dataset"])

    for day in range(1, 8):
        temp_left = pd.read_csv(
            f"results/result-{case_name}-{day}-day-{noise_percent}-noise-{run_count_left}-run.csv",
            index_col=0
        )

        temp_right = pd.read_csv(
            f"results/result-{case_name}-{day}-day-{noise_percent}-noise-{run_count_right}-run.csv",
            index_col=0
        )

        frame_left = pd.DataFrame(data={
            "accuracy": temp_left[f"accuracy_{noise_name}"],
            "day": day,
            "dataset": f"{run_count_left}-run"
        })

        frame_right = pd.DataFrame(data={
            "accuracy": temp_right[f"accuracy_{noise_name}"],
            "day": day,
            "dataset": f"{run_count_right}-run"
        })

        results = pd.concat([results, frame_left, frame_right], axis=0)

    results = results.astype({"accuracy": float, "day": int, "dataset": str})

    #***********************************************************************************************
    # Set AeStHeTiCs.
    #***********************************************************************************************
    sns.set_theme(style="whitegrid")

    #fontsize = 40
    #legend_fontsize = 20
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
    # Plot the results.
    #***********************************************************************************************
    (fig, axs) = plt.subplots()

    fig.tight_layout()

    sns.violinplot(
        x="day", y="accuracy", hue="dataset", data=results,
        color=".8", dodge=True, inner=None, ax=axs
    )

    sns.stripplot(
        x="day", y="accuracy", hue="dataset", data=results,
        alpha=0.3, ax=axs, dodge=True, linewidth=1, size=5
    )

    axs.set_ylabel("Accuracy")
    axs.set_xlabel("Window Width (Days)")

    axs.set_yticks([round(0.1 * i, 2) for i in range(3, 11)])

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
