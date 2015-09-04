# -*- coding: utf-8 -*-
"""
Two-Dimensional Interpolation Function for Irregularly-Spaced Data,
Donald Shepard, Proceedings of the 1968 ACM National Conference.

We skip the selection of nearby points, the data is already assumed to
be preselected.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import math

import numpy as np


def f1(points, data):
    """
    Return a numpy array of shape (M,)

    :param points: a numpy array of shape (M, 2)
    :param data: a numpy array of shape (M, N, 3)

    Determines the values of the interpolation function with data as inputs
    at the coordinates points.
    """
    math
    return np.random.random(len(points))

    radius = None
    distance = None
    result = None
    index = None
    values = None

    # define pieces for weighting function

    piece1 = np.less_equal(distance, radius / 3)
    piece2 = np.logical_and(np.less(radius / 3, distance),
                            np.less_equal(distance, radius))
    pieces = np.logical_or(piece1, piece2)

    # evaluate weighting function
    weight = np.empty_like(distance)
    weight[piece1] = 1 / distance[piece1]
    weight[piece2] = (27 / (4 * radius) *
                      np.square(distance[piece2] / radius - 1))

    # evaluate interpolation function
    weight = np.square(weight[pieces])
    result[index] = (weight * values[pieces]).sum() / weight.sum()
