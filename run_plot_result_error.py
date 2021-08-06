# SPDX-License-Identifier: MIT

"""
This script plots the frobenius error results.
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
    noise_percent = "0p005"

    #***********************************************************************************************
    # Load the results and place them in long form.
    #***********************************************************************************************
    result = pd.DataFrame(columns=["accuracy", "day", "dataset"])

    for day in range(1, 8):
        temp = pd.read_csv(f"results/result-{day}-day-{noise_percent}-noise.csv", index_col=0)

        noise = pd.DataFrame(data={
            "error": temp["noise_error"],
            "day": day,
            "dataset": "noisy"
        })

        denoise = pd.DataFrame(data={
            "error": temp["denoise_error"],
            "day": day,
            "dataset": "denoised"
        })

        result = pd.concat([result, noise, denoise], axis=0)

    #***********************************************************************************************
    # Set font sizes.
    #***********************************************************************************************
    sns.set_theme(style="whitegrid")

    #fontsize = 30
    #plt.rc("axes", labelsize=fontsize)
    #plt.rc("legend", title_fontsize=fontsize)
    #plt.rc("legend", fontsize=fontsize)
    #plt.rc("xtick", labelsize=fontsize)
    #plt.rc("ytick", labelsize=fontsize)

    #***********************************************************************************************
    # Plot the results.
    #***********************************************************************************************
    (fig, axs) = plt.subplots()

    fig.tight_layout()

    sns.violinplot(
        x="day", y="error", hue="dataset", data=result,
        color=".8", dodge=True, inner=None, ax=axs
    )

    sns.stripplot(
        x="day", y="error", hue="dataset", data=result,
        alpha=0.3, ax=axs, dodge=True, linewidth=1, size=5
    )

    axs.set_ylabel("Frobenius Error")
    axs.set_xlabel("Window Width (Days)")

    plt.show()

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
