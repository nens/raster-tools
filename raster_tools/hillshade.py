# -*- coding: utf-8 -*-
"""
Hillshade a la gdaldem.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import math
import os
import sys

import numpy as np

from raster_tools import gdal
from raster_tools import datasets
from raster_tools import groups
from raster_tools import utils

logger = logging.getLogger(__name__)
driver = gdal.GetDriverByName(str('gtiff'))


def process(array, values=16):
    zz = 2 * 2
    alt = math.radians(45)
    az = math.radians(315)

    y = np.empty_like(array)
    y[0] = (y[1] - y[0]) * 2
    y[-1] = (y[-1] - y[-2]) * 2
    y[1: -1] = y[2:] - y[:-2]

    x = np.empty_like(array)
    x[:, 0] = (x[:, 1] - x[:, 0]) * 2
    x[:, -1] = (x[:, -1] - x[:, -2]) * 2
    x[:, 1: -1] = x[:, 2:] - x[:, :-2]

    xx_plus_yy = x * x + y * y

    aspect = np.arctan2(y, x)
    cang = (math.sin(alt) -
            math.cos(alt * zz) * np.sqrt(xx_plus_yy) *
            np.sin(aspect - az)) / np.sqrt(1 + zz * xx_plus_yy)
    result = np.where(cang < 0, 1, 1 + 254 * cang).astype('u1')
    return result

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
    # savefig('hillshade.png')
    # print(np.unique(result))
    return result


class Calculator(object):
    def __init__(self, raster_path, output_path):
        self.group = groups.Group(gdal.Open(raster_path))
        self.output_path = output_path

    def calculate(self, feature):
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
        array = process(self.group.read(indices))

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


def hillshade(index_path, raster_path, output_path, part):
    """ Convert all features. """
    index = utils.PartialDataSource(index_path)
    if part is not None:
        index = index.select(part)

    calculator = Calculator(raster_path=raster_path, output_path=output_path)

    for feature in index:
        calculator.calculate(feature)
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
    """ Call hillshade with args from parser. """
    # logging
    kwargs = vars(get_parser().parse_args())
    if kwargs.pop('verbose'):
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level, format='%(message)s')

    # run or fail
    try:
        hillshade(**kwargs)
        return 0
    except:
        logger.exception('An exception has occurred.')
        return 1
