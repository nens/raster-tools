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


class Interpolator(object):

    def __init__(self, target_points, source_points, source_values, radius):
        """ Calculations that all interpolations use. """
        # calculate distance
        vectors = source_points[np.newaxis] - target_points[:, np.newaxis]
        distance = np.sqrt(np.square(vectors).sum(2))

        # calculate weight pieces
        piece1 = np.less_equal(distance, radius / 3)
        piece3 = np.greater(distance, radius)
        piece2 = ~np.logical_or(piece1, piece3)

        # evaluate weights per piece
        weights = np.empty_like(distance)
        weights[piece1] = 1 / distance[piece1]
        weights[piece2] = (27 / (4 * radius) *
                           np.square(distance[piece2] / radius - 1))
        weights[piece3] = 0

        # export
        self.target_points = target_points
        self.source_points = source_points
        self.source_values = source_values

        self.di = distance
        self.si = weights

    def interpolate_f2(self):
        """ Called f2 in the paper. """
        values = self.source_values[np.newaxis]
        sum_of_weighted_values = (np.square(self.si) * values).sum(1)
        sum_of_weights = np.square(self.si).sum(1)

        return sum_of_weighted_values / sum_of_weights

    def interpolate_f3(self):
        """ Called f3 in the paper. """
        xi = self.source_points[:, :1].reshape(1, -1, 1)  # source x
        yi = self.source_points[:, 1:].reshape(1, -1, 1)  # source y
        xj = self.source_points[:, :1].reshape(1, 1, -1)  # source x
        yj = self.source_points[:, 1:].reshape(1, 1, -1)  # source y

        p = self.target_points[:, 1:].reshape(-1, 1, 1)  # target x
        q = self.target_points[:, 1:].reshape(-1, 1, 1)  # target y

        di_dj = self.di[:, np.newaxis] * self.di[:, :, np.newaxis]
        cosine = ((p - xi) * (p - xj) + (q - yi) * (q - yj)) / di_dj
        si = self.si[:, np.newaxis]

        sum_of_weighted_weights = (si * (1 - cosine)).sum(2)
        sum_of_weights = si.sum(2)
        ti = sum_of_weighted_weights / sum_of_weights

        wi = np.square(self.si) * (1 + ti)

        values = self.source_values[np.newaxis]
        sum_of_weighted_values = (wi * values).sum(1)
        sum_of_weights = wi.sum(1)
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
    interpolator = Interpolator(target_points=target_points,
                                source_points=source_points,
                                source_values=source_values, radius=radius)
    return interpolator.interpolate_f3()
