# -*- coding: utf-8 -*-
"""
Two-Dimensional Interpolation Function for Irregularly-Spaced Data,
Donald Shepard, Proceedings of the 1968 ACM National Conference.

We skip the selection of nearby points, the data is already assumed to
be preselected.

Furthermore

options:
- pick a random number
- pick educated kdtree query based on maximum free path
- use linear rbf


Select targets within a radius
Select Sources within a radius
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import numpy as np


def weigh_s(distance, radius):
    """ Called s in the paper. """
    # define pieces for weighting function
    piece1 = np.less_equal(distance, radius / 3)
    piece3 = np.greater(distance, radius)
    piece2 = np.logical_xor(piece1, piece3)

    # evaluate weighting function
    weight = np.empty_like(distance)
    weight[piece1] = 1 / distance[piece1]
    weight[piece2] = (27 / (4 * radius) *
                      np.square(distance[piece2] / radius - 1))
    weight[piece3] = 0

    return weight


def interpolate_f2(target_points, source_points, source_values, radius):
    """ Called f2 in the paper. """
    source_count = len(source_points)
    source_indices = np.arange(len(source_points))

    # np.random.seed(0)
    np.random.shuffle(source_indices)

    target_count = len(target_points)
    batch_size = 5000000 // target_count

    sum_of_weights = np.zeros(target_count)
    sum_of_weighted_values = np.zeros(target_count)

    for i in xrange(0, source_count, batch_size):
        batch = slice(i * batch_size, i * batch_size + batch_size)

        # the weights
        source_points_batch = source_points[batch][:, np.newaxis, :]
        vector_batch = source_points_batch - target_points[np.newaxis, :]
        distance_batch = np.sqrt(np.square(vector_batch).sum(2))
        weights_batch = weigh_s(distance=distance_batch, radius=radius)

        # add to the sums
        source_values_batch = source_values[batch][:, np.newaxis]
        sum_of_weighted_values += (weights_batch * source_values_batch).sum(0)
        sum_of_weights += weights_batch.sum(0)

    return sum_of_weighted_values / sum_of_weights


def interpolate(target_points, source_points, source_values, radius):
    """
    Return a numpy array of shape (M,)

    :param target_points: a numpy array of shape (M, 2)
    :param source_points: a numpy array of shape (N, 2)
    :param source_values: a numpy array of shape (N,)

    Determines the values of the interpolation function with data as inputs
    at the coordinates points.
    """
    return interpolate_f2(target_points=target_points,
                          source_points=source_points,
                          source_values=source_values, radius=radius)
