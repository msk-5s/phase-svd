# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains a factory for making plots.
"""

from typing import Any, Optional, Tuple
from nptyping import NDArray

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_correlation_heatmap(
    data: NDArray[(Any, Any), float], labels: NDArray[(Any,), int], aspect: str = "equal"
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Make a heatmap of the correlations between loads.

    Parameters
    ----------
    data : numpy.ndarray, (n_timestep, n_load)
        The load data to plot.
    labels : numpy.ndarray, (n_load,)
        The phase labels of the loads.
    aspect : str, default="equal", ["auto", "equal"]
        The aspect mode to use.
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

    #***********************************************************************************************
    # Create the dotted boundaries to seperated the loads by phase more distinctly.
    # We want to place the phase labels inbetween the boundaries.
    #***********************************************************************************************
    # Boundaries to show the phases more distinctly using horizontal and vertical lines.
    phase_counts = np.bincount(labels)
    boundaries = [phase_counts[0], phase_counts[0] + phase_counts[1]]

    # Tick positions for the phases in the graph axis. This allows us to position the labels "A",
    # "B", etc. inbetween each boundary lines.
    tick_positions = [
        boundaries[0] // 2,
        boundaries[0] + ((boundaries[1] - boundaries[0]) // 2),
        boundaries[1] + ((len(labels) - boundaries[1]) // 2)
    ]

    tick_labels = ["A", "B", "C"]

    #***********************************************************************************************
    # Make the heatmap and add a color bar on the side to show the correlation value/color
    # relationship.
    #***********************************************************************************************
    (figure, axs) = plt.subplots()

    cor = np.corrcoef(data[:, sort_indices], rowvar=False)

    image = axs.imshow(cor, aspect=aspect)

    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=-1, vmax=1), cmap=image.cmap
    )

    cbar = axs.figure.colorbar(mappable=mappable, ax=axs)
    cbar.ax.set_ylabel("Coefficient")

    #***********************************************************************************************
    # Label the plot's axis' and plot the boundary lines.
    #***********************************************************************************************
    figure.tight_layout()
    axs.set_xlabel("Load")
    axs.set_ylabel("Load")
    axs.set_xticks(tick_positions)
    axs.set_yticks(tick_positions)
    axs.set_xticklabels(tick_labels)
    axs.set_yticklabels(tick_labels)

    for boundary in boundaries:
        axs.axhline(y=boundary, color="red", linestyle="dashed", linewidth=5)
        axs.axvline(x=boundary, color="red", linestyle="dashed", linewidth=5)

    return (figure, axs)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_value_plot(
    values: NDArray[(Any,), float], value_count: int, size: Optional[float] = None
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Make a plot of singular values.

    Parameters
    ----------
    values : numpy.ndarray, ndim=1, (min(n_load, n_timestep),)
        The singular values to plot.
    value_count : int
        The number of singular values that are being kept for data reconstruction.
    size : optional of int, default=None
        The size of the points in the scatter plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The plot figure.
    axs : matplotlib.axes.Axes
        The axis of the plot figure.
    """
    fig, axs = plt.subplots()

    fig.tight_layout()
    axs.set_xlabel("Singular Value #")
    axs.set_ylabel("log(Value)")

    axs.scatter(np.arange(len(values)) + 1, np.log(values), s=size)
    axs.axvline(value_count, linestyle="dashed", color="red")

    return (fig, axs)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_voltage_series_iqr_plot(
    series_n: NDArray[(Any,), float], series_nf: NDArray[(Any,), float],
    thresholds: Optional[Tuple[float, float]] = None
)-> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Make a plot of a voltage time series that is filtered with the IQR detector.

    Parameters
    ----------
    series_n : numpy.ndarray of float, (n_timestep,)
        The noisy time series.
    series_nf : numpy.ndarray of float, (n_timestep,)
        The IQR filtered time series.
    thresholds : optional of tuple of (float, float), default=None
        The upper and lower IQR thresholds, respectively. `None` will draw no threshold lines.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The plot figure.
    axs : matplotlib.axes.Axes
        The axis of the plot figure.
    """
    fig, axs = plt.subplots()
    fig.tight_layout()

    axs.plot(series_n, "r--", label="Noisy")
    axs.plot(series_nf, "k", label="IQR Filtered")

    if thresholds is not None:
        axs.axhline(y=thresholds[0], linestyle="dashed", color="blue")
        axs.axhline(y=thresholds[1], linestyle="dashed", color="blue")

    axs.set_xlabel("Time (15-min)")
    axs.set_ylabel("Voltage (V)")

    axs.legend()

    return (fig, axs)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def make_spectrogram_plot(
    windows: NDArray[(Any, Any), float], cutoff: Optional[float] = None
)-> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Make a spectrogram plot from a matrix of windows of a time series.

    Parameters
    ----------
    windows : numpy.ndarray, (width, n_window)
        The windows of a time series whose spectrogram will be plotted.
    cutoff : optional of float, default=None
        The cutoff frequency. A dashed horizontal line will be drawn at the cutoff frequency.

    Returns
    -------
    figure : matplotlib.figure.Figure
        The plot figure.
    axs : matplotlib.axes.Axes
        The axis of the plot figure.
    """
    #***********************************************************************************************
    # Calculate the DFT of each window.
    #***********************************************************************************************
    magnitudes_db = np.zeros(shape=windows.shape)

    for (i, window) in enumerate(windows.T):
        window_dft = np.fft.fftshift(np.fft.fft(window))

        # Add a small number (1e-6) to prevent log of zero.
        magnitudes_db[:, i] = 10 * np.log10(np.abs(window_dft) + 1e-6)

    #***********************************************************************************************
    # Plot the periodogram of each window to get the spectrogram.
    #***********************************************************************************************
    (fig, axs) = plt.subplots()

    fig.tight_layout()

    image = axs.imshow(
        magnitudes_db, cmap="seismic", aspect="auto", interpolation="none",
        extent=[1, windows.shape[1], 0.5, -0.5]
    )

    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=np.min(magnitudes_db), vmax=np.max(magnitudes_db)),
        cmap=image.cmap
    )

    cbar = axs.figure.colorbar(mappable=mappable, ax=axs)
    cbar.ax.set_ylabel("(dbV)")

    if cutoff is not None:
        color = (0.0, 0.5, 0.0)

        axs.axhline(y=cutoff, color=color, linestyle="dashed", linewidth=3)
        axs.axhline(y=-float(cutoff), color=color, linestyle="dashed", linewidth=3)

    axs.set_xlabel(f"Window ({windows.shape[0]} timestep/window)")
    axs.set_ylabel(r"$\omega$ (cycles/sample)")

    #***********************************************************************************************
    # Set the y-ticks.
    #***********************************************************************************************
    axs.set_yticks(np.arange(start=-0.5, stop=0.55, step=0.1))

    return (fig, axs)
