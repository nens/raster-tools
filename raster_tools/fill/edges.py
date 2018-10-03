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
        :param indices: tuple of indices
        :param values: values
        :param rows: first axis indices
        :param cols: second axis indices
        """
        self.indices = indices
        self.values = values
        self.shape = shape

        self.full = len(values) == self.shape[0] * self.shape[1]

    def aggregated(self):
        """ Return aggregated edge object. """
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
        """ Return fresh array. """
        array = np.full(self.shape, FILLVALUE, dtype=DTYPE)
        self.pasteon(array)
        return array
