# -*- coding: utf-8 -*-
"""
Thin bytes in a dataset of type byte (for example from gdaldem hillshade),
using local statistics. This improves the compression of the dataset.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import os

import numpy as np

from raster_tools import gdal
from raster_tools import datasets
from raster_tools import groups
from raster_tools import utils

logger = logging.getLogger(__name__)
driver = gdal.GetDriverByName(str('gtiff'))


def process(array, values=16):
    # analyze data
    lo, hi = array.min(), array.max()
    bins = np.mgrid[lo:hi:257j]  # if applied to dem data, probably need more
    histogram = np.hstack([[0], np.histogram(array, bins)[0]])

    # make interpolation to determine levels
    stat_x = histogram.cumsum() / histogram.sum() * (hi - lo) + lo
    stat_y = bins
    edges = np.mgrid[lo:hi:(1 + values) * 1j]
    centers = (edges[1:] + edges[:-1]) / 2
    levels = np.interp(centers, stat_x, stat_y)

    # make interpolation to apply to data
    thin_y = np.sort(np.hstack(2 * [levels]))
    thin_x = np.sort(np.hstack([edges[1:], edges[:-1]]))
    result = np.interp(array, thin_x, thin_y).astype(array.dtype)
    # from pylab import plot, savefig, show
    # plot(stat_x, stat_y)
    # plot(array.flatten(), result.flatten(), '.')
    # plot(centers, levels, 'o')
    # plot(thin_x, thin_y, 'o')
    # show()
    # savefig('thinbytes.png')
    # print(np.unique(result))
    return result


class Converter(object):
    def __init__(self, raster_path, output_path):
        self.group = groups.Group(gdal.Open(raster_path))
        self.output_path = output_path

    def convert(self, feature):
        # target path
        leaf_number = feature[b'BLADNR']
        path = os.path.join(self.output_path,
                            leaf_number[:3],
                            '{}.tif'.format(leaf_number))
        if os.path.exists(path):
            logger.debug('Target already exists.')
            return

        # calculate
        geometry = feature.geometry()
        indices = self.group.geo_transform.get_indices(geometry)
        array = self.group.read(indices)
        for i in range(0, 2500, 250):
            for j in range(0, 2000, 250):
                view = array[i:i + 250, j: j + 250]
                view[:] = process(array=view, values=32)

        # create directory
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass  # no problem

        # save
        kwargs = {
            'projection': self.group.projection,
            'geo_transform': self.group.geo_transform.shifted(geometry),
        }
        options = [
            'tiled=yes',
            'compress=deflate',
        ]
        with datasets.Dataset(array[np.newaxis, ...], **kwargs) as dataset:
            driver.CreateCopy(path, dataset, options=options)


def thinbytes(index_path, raster_path, output_path, part):
    """ Convert all features. """
    index = utils.PartialDataSource(index_path)
    if part is not None:
        index = index.select(part)

    converter = Converter(raster_path=raster_path, output_path=output_path)

    for feature in index:
        converter.convert(feature)
    return 0


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'index_path',
        metavar='INDEX',
        help='shapefile with geometries and names of output tiles',
    )
    parser.add_argument(
        'raster_path',
        metavar='HILLSHADE',
        help='path to GDAL hillshade raster dataset.'
    )
    parser.add_argument(
        'output_path',
        metavar='IMAGES',
        help='target directory',
    )
    parser.add_argument(
        '-p', '--part',
        help='partial processing source, for example "2/3"',
    )
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


def main():
    """ Call thinbytes with args from parser. """
    # logging
    kwargs = vars(get_parser().parse_args())
    if kwargs.pop('verbose'):
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level, format='%(message)s')

    # run or fail
    try:
        thinbytes(**kwargs)
        return 0
    except:
        logger.exception('An exception has occurred.')
        return 1
