# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-
"""
Image utility to plot edges and arrays.
"""

from os.path import dirname
import os

import numpy as np

from raster_tools.fill import edges


class Imager(object):
    """
    Plot arrays or edge objects, only meant for debugging the algorithm.
    """
    def __init__(self):
        self.disable()

    def enable(self, path):  # pragma: no cover
        self.path = path
        self.debug = self._debug

    def disable(self):
        self.path = None
        self.debug = self._no_debug

    def _no_debug(self, obj, text):
        """ Does nothing. """
        pass

    def _debug(self, obj, text):  # pragma: no cover
        """ Saves a PNG image. """
        from pylab import imshow, plot, savefig, subplot2grid, title, xlim

        # convert edge to array
        if isinstance(obj, edges.Edge):
            obj = np.ma.masked_equal(obj.toarray(), edges.FILLVALUE)
            obj.data[obj.mask] = 0  # or matplotlib complains

        height, width = obj.shape

        # plot array
        subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2)
        imshow(obj)
        title(text)

        # horizontal samples at 1/3 and 2/3
        for i in (0, 1):
            subplot2grid((3, 3), (i, 2))
            index = (i + 1) * height // 3
            plot(obj[index], '.')
            xlim(-1, width)

        # vertical samples at 1/3 and 2/3
        for i in (0, 1):
            subplot2grid((3, 3), (2, i))
            index = (i + 1) * width // 3
            plot(obj[:, index], '.')
            xlim(-1, height)

        # topleft to bottomright sample
        subplot2grid((3, 3), (2, 2))
        size = max(width, height)
        index = (
            np.linspace(0.5, height - 0.5, size).astype('i8'),
            np.linspace(0.5, width - 0.5, size).astype('i8'),
        )
        plot(obj[index], '.')
        xlim(-1, height)

        # save
        target_path = '{}/{}.png'.format(
            self.path, text.lower().replace(' ', '_'),
        )

        try:
            os.makedirs(dirname(target_path))
        except OSError:
            pass
        savefig(target_path)
