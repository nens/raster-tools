# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-
"""
Edge class.
"""

import collections
import statistics

import numpy as np

# properties of working arrays
DTYPE = 'f4'
FILLVALUE = np.finfo(DTYPE).max


class Edge(object):
    def __init__(self, indices, values, shape):
        """
        An Edge object represents a contour of pixels in a raster image.

        :param indices: indices as a tuple of integer numpy arrays
        :param values: values of the pixels at indices
        :param shape: shape of the envelope containing the edge
        """
        self.indices = indices
        self.values = values
        self.shape = shape

    @property
    def is_full(self):
        return len(self.values) == self.shape[0] * self.shape[1]

    def aggregated(self):
        """
        Return aggregated edge object.

        The aggregated edge is a new edge object where the edge pixels are the
        median of up to four underlying pixels from self.
        """
        # aggregate
        work = collections.defaultdict(list)
        for k, i, j in zip(self.values, *self.indices):
            work[i // 2, j // 2].append(k)

        # statistic
        indices = tuple(np.array(ind) for ind in zip(*work))
        values = [statistics.median(work[k]) for k in zip(*indices)]
        return self.__class__(
            indices=indices,
            values=values,
            shape=(-(-self.shape[0] // 2), -(-self.shape[1] // 2)),
        )

    def pasteon(self, array):
        """ Paste values on array. """
        array[self.indices] = self.values

    def toarray(self):
        """ Convert this edge into an array. """
        array = np.full(self.shape, FILLVALUE, dtype=DTYPE)
        self.pasteon(array)
        return array
