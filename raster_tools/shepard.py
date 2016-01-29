# -*- coding: utf-8 -*-
"""
Two-Dimensional Interpolation Function for Irregularly-Spaced Data,
Donald Shepard, Proceedings of the 1968 ACM National Conference.

We skip the selection of nearby points, the data is already assumed to
be preselected.

After direction and sloping have been incorporated, let's revisit the
weighting function.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import numpy as np


def weigh_4(distance, radius):
    """ Debugging purposes. """
    return 1 / distance


def weigh_s(distance, radius):
    """ Called s in the paper. """
    # define pieces for weighting function
    piece1 = np.less_equal(distance, radius / 3)
    piece3 = np.greater(distance, radius)
    piece2 = ~np.logical_or(piece1, piece3)

    # evaluate weighting function
    weight = np.empty_like(distance)
    weight[piece1] = 1 / distance[piece1]
    weight[piece2] = (27 / (4 * radius) *
                      np.square(distance[piece2] / radius - 1))
    weight[piece3] = 0

    return weight


def interpolate_f2(target_points, source_points, source_values, radius):
    """ Called f2 in the paper. """

    # the weights
    vectors = source_points[:, np.newaxis] - target_points[np.newaxis]
    distance = np.sqrt(np.square(vectors).sum(2))
    weights = weigh_s(distance=distance, radius=radius)

    # add to the sums
    sum_of_weighted_values = (weights * source_values[:, np.newaxis]).sum(0)
    sum_of_weights = weights.sum(0)

    return sum_of_weighted_values / sum_of_weights


def interpolate_f3(target_points, source_points, source_values, radius):
    """ Called f3 in the paper. """
    if target_points.size < 10:
        return
    import ipdb
    ipdb.set_trace()


def interpolate(target_points, source_points, source_values, radius):
    """
    Return a numpy array of shape (M,)

    :param target_points: a numpy array of shape (M, 2)
    :param source_points: a numpy array of shape (N, 2)
    :param source_values: a numpy array of shape (N,)

    Determines the values of the interpolation function with data as inputs
    at the coordinates points.
    """
    return interpolate_f3(target_points=target_points,
                          source_points=source_points,
                          source_values=source_values, radius=radius)
