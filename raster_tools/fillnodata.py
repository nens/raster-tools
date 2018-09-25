# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-
"""
Filler.

The idea is to get a tension-like result, but much less computationally
intensive.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from os.path import dirname, exists

import argparse
import collections
import os

from osgeo import gdal
from scipy import ndimage
import numpy as np

from raster_tools import datasets

GTIF = gdal.GetDriverByName(str('gtiff'))
OPTIONS = ['compress=deflate', 'tiled=yes']
KERNEL = np.array([[0.0625, 0.1250,  0.0625],
                   [0.1250, 0.2500,  0.1250],
                   [0.0625, 0.1250,  0.0625]])


def smooth(array):
    """ Two-step uniform for symmetric smoothing. """
    return ndimage.correlate(array, KERNEL, output=array)


def zoom(array):
    """ Return zoomed array. """
    return array.repeat(2, axis=0).repeat(2, axis=1)


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
        total = collections.defaultdict(float)
        count = collections.defaultdict(float)
        for k, i, j in zip(self.values, *self.indices):
            total[i // 2, j // 2] += k
            count[i // 2, j // 2] += 1

        # statistic
        indices = tuple(np.array(ind) for ind in zip(*total))
        values = [total[k] / count[k] for k in zip(*indices)]
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
        array = np.empty(self.shape, 'f4')
        self.pasteon(array)
        return array


class Exchange(object):
    def __init__(self, source_path, target_path):
        """
        Read source, create target array.
        """
        dataset = gdal.Open(source_path)
        band = dataset.GetRasterBand(1)

        self.source = band.ReadAsArray()
        self.no_data_value = band.GetNoDataValue()

        self.mask = (self.source == self.no_data_value)
        self.shape = (self.source.shape)

        self.kwargs = {
            'no_data_value': self.no_data_value,
            'projection': dataset.GetProjection(),
            'geo_transform': dataset.GetGeoTransform(),
        }

        self.target_path = target_path
        self.target = np.full_like(self.source, self.no_data_value)

    def _grow(self, obj):
        """
        Return grown slices tuple, but not beyond our shape.

        :param obj: tuple of slices
        """
        return (
            slice(
                max(0, obj[0].start - 1),
                min(self.shape[0], obj[0].stop + 1),
            ),
            slice(
                max(0, obj[1].start - 1),
                min(self.shape[1], obj[1].stop + 1),
            ),
        )

    def __iter__(self):
        """
        Return generator of (source, target, void) tuples.

        Source and target are views into a larger array. Void is a newly
        created array containing the footprint of the void.
        """
        gdal.TermProgress_nocb(0)

        # analyze
        labels, total = ndimage.label(self.mask)
        items = ndimage.find_objects(labels)

        # iterate the objects
        for label, item in enumerate(items, 1):
            index = self._grow(item)       # to include the edge
            source = self.source[index]    # view into source array
            target = self.target[index]    # view into target array
            void = labels[index] == label  # the footprint of this void
            yield source, target, void

            gdal.TermProgress_nocb(label / total)

    def save(self):
        """ Save. """
        # prepare dirs
        try:
            os.makedirs(dirname(self.target_path))
        except OSError:
            pass

        # write tiff
        array = self.target[np.newaxis]
        with datasets.Dataset(array, **self.kwargs) as dataset:
            GTIF.CreateCopy(self.target_path, dataset, options=OPTIONS)


def fill(edge):
    """
    Return a filled array.

    :param edge: Edge instance.
    """
    # aggregate the edge
    aggregated = edge.aggregated()

    if aggregated.full:
        # convert the aggregated edge into an array
        array = aggregated.toarray()
    else:
        # fill the aggregated edge and return the array
        array = fill(aggregated)  # recursively fills

    array = zoom(array)[:edge.shape[0], :edge.shape[1]]
    edge.pasteon(array)
    smooth(array)
    return array


def fillnodata(source_path, target_path):
    """ Fill the voids in a single file. """
    # skip existing
    if exists(target_path):
        print('{} skipped.'.format(target_path))
        return

    # skip when missing sources
    if not exists(source_path):
        print('{} not found.'.format(source_path))
        return

    # read
    exchange = Exchange(source_path, target_path)

    # process
    for count, (source, target, void) in enumerate(exchange, 1):
        # analyze
        edge = void ^ ndimage.binary_dilation(void)
        indices = edge.nonzero()

        # fill
        edge = Edge(
            indices=indices,
            values=source[indices],
            shape=source.shape,
        )
        filled = fill(edge)

        # apply
        target[void] = filled[void]

    # save
    exchange.save()


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # positional arguments
    parser.add_argument(
        'source_path',
        metavar='SOURCE',
    )
    parser.add_argument(
        'target_path',
        metavar='TARGET',
    )

    return parser


def main():
    """ Call command with args from parser. """
    fillnodata(**vars(get_parser().parse_args()))
