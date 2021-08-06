# SPDX-License-Identifier: MIT

"""
This script generates summary statistics of the results.
"""

import pandas as pd

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def _make_stats_df(key: str, noise_percent: str) -> pd.DataFrame:
    """
    Make the descriptive statistics of the results.

    Parameters
    ----------
    key : str
        The key to use.
    noise_percent : str
        The noise percent of the results to use.

    Returns
    -------
    pandas.DataFrame
        The dataframe of descriptive statistics.
    """
    stats_df = pd.DataFrame(data={
        "Window": list(range(1, 8)),
        "Mean": 0.0,
        "Median": 0.0,
        "Std": 0.0,
        "Min": 0.0,
        "Max": 0.0,
        "Skewness": 0.0,
        "Kurtosis": 0.0
    })

    for day in range(1, 8):
        result_df = pd.read_csv(
            f"results/result-{day}-day-{noise_percent}-noise.csv", index_col=0
        )

        i = day - 1

        stats_df.at[i, "Mean"] = result_df[key].mean()
        stats_df.at[i, "Median"] = result_df[key].median()
        stats_df.at[i, "Std"] = result_df[key].std()
        stats_df.at[i, "Min"] = result_df[key].min()
        stats_df.at[i, "Max"] = result_df[key].max()
        stats_df.at[i, "Skewness"] = result_df[key].skew()
        stats_df.at[i, "Kurtosis"] = result_df[key].kurtosis()

    return stats_df

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def main(): # pylint: disable=too-many-locals, too-many-statements
    """
    The main function.
    """
    #***********************************************************************************************
    # Make the descriptive statistics.
    #***********************************************************************************************
    for noise_percent in ["0p001", "0p002", "0p005", "0p01"]:
        accuracy_noise_df = _make_stats_df(key="noise_accuracy", noise_percent=noise_percent)
        accuracy_denoise_df = _make_stats_df(key="denoise_accuracy", noise_percent=noise_percent)

        error_noise_df = _make_stats_df(key="noise_error", noise_percent=noise_percent)
        error_denoise_df = _make_stats_df(key="denoise_error", noise_percent=noise_percent)

        noise_str = str(noise_percent).replace(".", "p")

        accuracy_noise_df.to_csv(f"stats/stats-{noise_str}-noise-accuracy.csv")
        accuracy_denoise_df.to_csv(f"stats/stats-{noise_str}-denoise-accuracy.csv")
        error_noise_df.to_csv(f"stats/stats-{noise_str}-noise-error.csv")
        error_denoise_df.to_csv(f"stats/stats-{noise_str}-denoise-error.csv")

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
