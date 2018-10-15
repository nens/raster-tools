# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-
"""
Filler.

The idea is to get a tension-like result, but much less computationally
intensive.
"""

from os.path import dirname, exists

import argparse
import os

from osgeo import gdal
from osgeo import ogr
from scipy import ndimage

import numpy as np

from raster_tools import datasets
from raster_tools.fill import edges
from raster_tools.fill import imagers

# output driver and optinos
DRIVER = gdal.GetDriverByName('gtiff')
OPTIONS = ['compress=deflate', 'tiled=yes']

# smoothing kernel designed to have the effect of restoring features after
# aggregation and zooming
KERNEL = np.array([[0.0625, 0.1250, 0.0625],
                   [0.1250, 0.2500, 0.1250],
                   [0.0625, 0.1250, 0.0625]])

imager = imagers.Imager()
progress = True


def smooth(array):
    """ Two-step uniform for symmetric smoothing. """
    return ndimage.correlate(array, KERNEL, output=array)


def zoom(array):
    """ Return zoomed array. """
    return array.repeat(2, axis=0).repeat(2, axis=1)


class Exchange(object):
    def __init__(self, source_path, target_path):
        """
        Read source, create target array.
        """
        dataset = gdal.Open(source_path)
        band = dataset.GetRasterBand(1)

        self.source = band.ReadAsArray()
        self.no_data_value = band.GetNoDataValue()

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
        if progress:  # pragma: no cover
            gdal.TermProgress_nocb(0)

        # analyze
        mask = (self.source == self.no_data_value)
        labels, total = ndimage.label(mask)
        items = ndimage.find_objects(labels)

        # iterate the objects
        for label, item in enumerate(items, 1):
            index = self._grow(item)       # to include the edge
            source = self.source[index]    # view into source array
            target = self.target[index]    # view into target array
            void = labels[index] == label  # the footprint of this void
            yield source, target, void

            if progress:  # pragma: no cover
                gdal.TermProgress_nocb(label / total)

    def clip(self, path):
        """
        Clip using OGR source at path.

        Clip actually puts zeros in the source outside the clip layer,
        so that they are excluded from the fill process.
        """
        # create mask with ones
        mask = np.ones_like(self.source, dtype='b1')

        # rasterize data_source as zeros
        data_source = ogr.Open(path)
        array = mask[np.newaxis].view('u1')
        with datasets.Dataset(array, **self.kwargs) as dataset:
            for layer in data_source:
                gdal.RasterizeLayer(dataset, [1], layer, burn_values=[0])

        # fill source with zeros where mask contains ones
        self.source[mask] = 0

    def round(self, decimals):
        """ Round target. """
        active = self.target != self.no_data_value
        self.target[active] = self.target[active].round(decimals)

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
            DRIVER.CreateCopy(self.target_path, dataset, options=OPTIONS)


def fill(edge, level=0):
    """
    Return a filled array.

    :param edge: Edge instance.
    """
    imager.debug(edge, 'Edge {}'.format(level))

    # aggregate the edge
    aggregated = edge.aggregated()

    if aggregated.full:
        # convert the aggregated edge into an array
        array = aggregated.toarray()

        imager.debug(array, 'Edge {}'.format(level + 1))

    else:
        # fill the aggregated edge and return the array
        array = fill(aggregated, level + 1)  # recursively fills

    array = zoom(array)[:edge.shape[0], :edge.shape[1]]

    imager.debug(array, '{}C Zoomed'.format(level))

    edge.pasteon(array)
    imager.debug(array, '{}B Edge pasted'.format(level))

    smooth(array)
    imager.debug(array, '{}A Smoothed'.format(level))

    return array


def fillnodata(source_path, target_path, clip_path, decimals):
    """ Fill the voids in a single file. """
    # skip existing
    if exists(target_path):
        print('{} skipped.'.format(target_path))
        return

    # skip when missing sources
    if not exists(source_path):
        print('Raster source "{}" not found.'.format(source_path))
        return
    if clip_path and not exists(clip_path):
        print('Clip source "{}" not found.'.format(clip_path))
        return

    # read
    exchange = Exchange(source_path, target_path)

    if clip_path:
        exchange.clip(clip_path)

    # process
    for count, (source, target, void) in enumerate(exchange, 1):

        # analyze
        edge = void ^ ndimage.binary_dilation(void)
        indices = edge.nonzero()

        # create edge object
        edge = edges.Edge(
            indices=indices,
            values=source[indices],
            shape=source.shape,
        )

        # fill it
        filled = fill(edge)

        # apply
        target[void] = filled[void]

    if decimals:
        exchange.round(decimals)

    # save
    exchange.save()


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__,
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
    parser.add_argument(
        '-r', '--round',
        type=int,
        dest='decimals',
        help='Round the result to this number of decimals.',
    )
    parser.add_argument(
        '-c', '--clip',
        dest='clip_path',
        help='Clip the result using this OGR data source.',
    )

    return parser


def main():
    """ Call command with args from parser. """
    fillnodata(**vars(get_parser().parse_args()))
