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

from raster_tools import datasets
from raster_tools import datasources
from raster_tools import gdal
from raster_tools import groups

logger = logging.getLogger(__name__)
driver = gdal.GetDriverByName(str('gtiff'))


def zevenbergen_thorne(array, resolution, altitude=45, azimuth=315):
    xres, yres = resolution
    alt = math.radians(altitude)
    az = math.radians(azimuth)
    zsf = 1 / 2
    square_zsf = zsf * zsf

    # gradient
    y = np.empty_like(array)
    y[1: -1] = (array[:-2] - array[2:]) / -yres
    y[-1] = (array[-2] - array[-1]) * 2 / -yres
    y[0] = (array[0] - array[1]) * 2 / -yres

    x = np.empty_like(array)
    x[:, 1: -1] = (array[:, :-2] - array[:, 2:]) / xres
    x[:, -1] = (array[:, -2] - array[:, -1]) * 2 / xres
    x[:, 0] = (array[:, 0] - array[:, 1]) * 2 / xres

    xx_plus_yy = x * x + y * y
    aspect = np.arctan2(y, x)

    cang = (math.sin(alt) -
            math.cos(alt) * zsf * np.sqrt(xx_plus_yy) *
            np.sin(aspect - az)) / np.sqrt(1 + square_zsf * xx_plus_yy)

    return np.where(cang <= 0, 1, 1 + 254 * cang).astype('u1')


def other(array, resolution, altitude=45, azimuth=315):
    xres, yres = resolution
    alt = math.radians(altitude)
    az = math.radians(azimuth)
    zsf = 1 / 8
    square_zsf = zsf * zsf

    # gradient
    s0 = slice(None, -2), slice(None, -2)
    s1 = slice(None, -2), slice(1, -1)
    s2 = slice(None, -2), slice(2, None)
    s3 = slice(1, -1), slice(None, -2)
    s4 = slice(1, -1), slice(1, -1)
    s5 = slice(1, -1), slice(2, None)
    s6 = slice(2, None), slice(None, -2)
    s7 = slice(2, None), slice(1, -1)
    s8 = slice(2, None), slice(2, None)

    y = np.empty_like(array)
    y[s4] = (array[s0] + 2 * array[s1] + array[s2]
             - array[s6] - 2 * array[s7] - array[s8]) / -yres

    x = np.empty_like(array)
    x[s4] = (array[s0] + 2 * array[s3] + array[s6]
             - array[s2] - 2 * array[s5] - array[s8]) / xres

    # TODO Edges
    x[0] = x[-1] = y[0] = y[-1] = 0
    x[:, 0] = x[:, -1] = y[:, 0] = y[:, -1] = 0
    x[0, 0] = x[-1, 0] = x[0, -1] = x[-1, -1] = 0
    y[0, 0] = y[-1, 0] = y[0, -1] = y[-1, -1] = 0

    xx_plus_yy = x * x + y * y
    aspect = np.arctan2(y, x)

    cang = (math.sin(alt) -
            math.cos(alt) * zsf * np.sqrt(xx_plus_yy) *
            np.sin(aspect - az)) / np.sqrt(1 + square_zsf * xx_plus_yy)

    return np.where(cang <= 0, 1, 1 + 254 * cang).astype('u1')


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
        array = self.group.read(indices)
        resolution = self.group.geo_transform[1], self.group.geo_transform[5]
        # hillshade = zevenbergen_thorne(array=array, resolution=resolution)
        hillshade = other(array=array, resolution=resolution)

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
        with datasets.Dataset(hillshade[np.newaxis, ...], **kwargs) as dataset:
            driver.CreateCopy(path, dataset, options=options)


def hillshade(index_path, raster_path, output_path, part):
    """ Convert all features. """
    index = datasources.PartialDataSource(index_path)
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
