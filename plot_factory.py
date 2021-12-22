# SPDX-License-Identifier: MIT

"""
This module contains a factory for making plots.
"""

from typing import Any, Optional, Tuple
from nptyping import NDArray

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.graphics.tsaplots

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_acf_plot(series: NDArray[(Any,), float], lags: Optional[int] = None)\
-> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Make an auto correlation function plot of a time series.

    Parameters
    ----------
    series : numpy.ndarray of float, (n_timestep,)
        The time series to plot.
    lags : optional of int, default=None
        The number of lags to include.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The plot figure.
    axs : matplotlib.axes.Axes
        The axis of the plot figure.
    """
    fig = statsmodels.graphics.tsaplots.plot_acf(x=series, lags=lags)
    axs = fig.gca()

    fig.tight_layout()
    axs.set_title(None)
    axs.set_xlabel("Time Lag (15-min)")
    axs.set_ylabel("Correlation")

    return (fig, axs)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_correlation_heatmap(load: NDArray[(Any, Any), float], labels: NDArray[(Any,), int])\
-> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Make a heatmap of the correlations between loads.

    Parameters
    ----------
    load : numpy.ndarray, (n_timestep, n_load)
        The load data to plot.
    labels : numpy.ndarray, (n_load,)
        The phase labels of the loads.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The plot figure.
    axs : matplotlib.axes.Axes
        The axis of the plot figure.
    """
    # Sort the loads by phase so we can better see how the phases follow the correlation structure
    # of the voltage measurements.
    sort_indices = np.argsort(labels)

    # Boundaries to show the phases more distinctly using horizontal and vertical lines.
    phase_counts = np.bincount(labels)
    boundaries = [phase_counts[0], phase_counts[0] + phase_counts[1]]

    # Tick positions for the phases in the graph axis.
    tick_positions = [
        boundaries[0] // 2,
        boundaries[0] + ((boundaries[1] - boundaries[0]) // 2),
        boundaries[1] + ((len(labels) - boundaries[1]) // 2)
    ]

    tick_labels = ["A", "B", "C"]

    (figure, axs) = plt.subplots()

    cor = np.corrcoef(load[:, sort_indices], rowvar=False)

    image = axs.imshow(cor, aspect="auto")

    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=-1, vmax=1), cmap=image.cmap
    )

    cbar = axs.figure.colorbar(mappable=mappable, ax=axs)
    cbar.ax.set_ylabel("Coefficient")

    figure.tight_layout()
    axs.set_xlabel("Load")
    axs.set_ylabel("Load")
    axs.set_xticks(tick_positions)
    axs.set_yticks(tick_positions)
    axs.set_xticklabels(tick_labels)
    axs.set_yticklabels(tick_labels)

    for boundary in boundaries:
        axs.axhline(y=boundary, color="red", linestyle="dashed")
        axs.axvline(x=boundary, color="red", linestyle="dashed")

    return (figure, axs)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_value_plot(
    singular_values: NDArray[(Any,), float], value_count: int, size: Optional[float] = None)\
    -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Make a plot of singular values.

    Parameters
    ----------
    singular_values : numpy.ndarray, ndim=1, (min(n_load, n_timestep),)
        The singular values to plot.
    value_count : int
        The number of singular values that are being kept for data reconstruction.
    size : optional of int, default=None
        The size of the points in the scatter plot.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The plot figure.
    axs : matplotlib.axes.Axes
        The axis of the plot figure.
    """
    figure, axs = plt.subplots()

    figure.tight_layout()
    axs.set_xlabel("Singular Value #")
    axs.set_ylabel("log(Value)")

    axs.scatter(np.arange(len(singular_values)) + 1, np.log(singular_values), s=size)
    axs.axvline(value_count, linestyle="dashed", color="red")

    return (figure, axs)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_voltage_series_plot(series: NDArray[(Any,), float])\
-> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Make a plot of a voltage time series.

    Parameters
    ----------
    series : numpy.ndarray of float, (n_timestep,)
        The time series to plot.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The plot figure.
    axs : matplotlib.axes.Axes
        The axis of the plot figure.
    """
    fig, axs = plt.subplots()

    fig.tight_layout()
    axs.plot(series)
    axs.set_xlabel("Time (15-min)")
    axs.set_ylabel("Voltage (V)")

    return (fig, axs)
