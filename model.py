# SPDX-License-Identifier: MIT

"""
This module contains functions for performing phase label correction.
"""

from typing import Any
from nptyping import NDArray

import numpy as np
import sklearn_extra.cluster

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def predict(
    labels: NDArray[(Any,), int], load: NDArray[(Any, Any), float], random_seed: int
) -> NDArray[(Any, Any), int]:
    """
    Predict the phase labels using clustering.

    Parameters
    ----------
    labels : numpy.ndarray of int, (n_load,)
        The true phase labels.
    load : numpy.ndarray of int, (n_timestep, n_load)
        The load voltage magnitude data.
    random_seed : int
        The random seed to use.

    Returns
    -------
    numpy.ndarray of int, (n_load,)
        The predicted phase labels.
    """
    dist = np.sqrt((1 - np.corrcoef(load, rowvar=False)) / 2)

    clusters = sklearn_extra.cluster.KMedoids(
        n_clusters=3, method="alternate", metric="precomputed", init="k-medoids++",
        random_state=random_seed
    ).fit_predict(dist)

    predictions = _predict_majority_vote(clusters=clusters, labels=labels)

    return predictions

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def _predict_majority_vote(clusters: NDArray[(Any,), int], labels: NDArray[(Any,), int])\
-> NDArray[(Any,), int]:
    """
    Do label correction using predicted clusters and labels with a majority vote rule.

    Parameters
    ----------
    clusters : numpy.ndarray of int, (n_load,)
        The predicted clusters for each load.
    labels : numpy.ndarray of int, (n_load,)
        The labels to use for the loads. These labels will be used in the majority
        vote approach.

    Returns
    -------
    numpy.ndarray of int, (n_load,)
        The label predictions via majority vote.
    """
    unique_clusters = np.unique(clusters)

    indices_list = [np.where(clusters == i)[0] for i in unique_clusters]

    predictions = np.zeros(shape=len(clusters), dtype=int)

    for indices in indices_list:
        observed_labels = labels[indices]
        predicted_label = np.bincount(observed_labels).argmax()

        predictions[indices] = predicted_label

    return predictions
