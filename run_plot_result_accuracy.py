# SPDX-License-Identifier: BSD-3-Clause

"""
This script plots the accuracy results.
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
    case_name = "case_svd_svht"
    noise_name = "gauss"
    noise_percent = "0p005"
    run_count = 10

    #***********************************************************************************************
    # Load the results and place them in long form.
    #***********************************************************************************************
    results = pd.DataFrame(columns=["accuracy", "day", "dataset"])

    for day in range(1, 8):
        temp_n = pd.read_csv(
            f"results/result-case_null-{day}-day-{noise_percent}-noise-{run_count}-run.csv",
            index_col=0
        )

        temp_nd = pd.read_csv(
            f"results/result-{case_name}-{day}-day-{noise_percent}-noise-{run_count}-run.csv",
            index_col=0
        )

        frame_n = pd.DataFrame(data={
            "accuracy": temp_n[f"accuracy_{noise_name}"],
            "day": day,
            "dataset": "noisy"
        })

        frame_nd = pd.DataFrame(data={
            "accuracy": temp_nd[f"accuracy_{noise_name}"],
            "day": day,
            "dataset": "denoised"
        })

        results = pd.concat([results, frame_n, frame_nd], axis=0)

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
