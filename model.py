# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains functions for performing phase label correction.
"""

from typing import Any
from nptyping import NDArray

import numpy as np
import sklearn.metrics
import sklearn_extra.cluster

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def predict(
    data: NDArray[(Any, Any), float], labels: NDArray[(Any,), int], random_state: int,
    run_count: int
) -> NDArray[(Any, Any), int]:
    """
    Predict the phase labels using clustering.

    Parameters
    ----------
    data : numpy.ndarray of int, (n_timestep, n_load)
        The load voltage magnitude data.
    labels : numpy.ndarray of int, (n_load,)
        The phase labels.
    random_state : int
        The random state to use.
    run_count : int
        The number of times to run the clustering.

    Returns
    -------
    numpy.ndarray of int, (n_load,)
        The predicted phase labels.
    """
    rng = np.random.default_rng(seed=random_state)

    dist = np.sqrt((1 - np.corrcoef(data, rowvar=False)) / 2)

    clusters_list = []
    scores = np.zeros(shape=run_count)

    for i in range(run_count):
        random_state_cluster = rng.integers(np.iinfo(np.int32).max)

        model = sklearn_extra.cluster.KMedoids(
            n_clusters=3, method="alternate", metric="precomputed", init="k-medoids++",
            random_state=random_state_cluster
        ).fit(dist)

        clusters = model.predict(dist)

        clusters_list.append(clusters)
        scores[i] = sklearn.metrics.calinski_harabasz_score(X=data.T, labels=clusters)

    # Use the clustering with the highest CH score.
    index = np.argmax(scores)
    clusters = clusters_list[index]

    predictions = _predict_majority_vote(clusters=clusters, labels=labels)

    return predictions

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def _predict_majority_vote(
    clusters: NDArray[(Any,), int], labels: NDArray[(Any,), int]
)-> NDArray[(Any,), int]:
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
