#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create tif rasters from xyz files by linear interpolation using griddata.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import math
import os

import numpy as np

from raster_tools import datasets
from raster_tools import gdal
from raster_tools import osr

WIDTH = 0.5
HEIGHT = 0.5
NO_DATA_VALUE = np.finfo('f4').min.item()
DRIVER = gdal.GetDriverByName(str('gtiff'))
OPTIONS = ['compress=deflate', 'tiled=yes']
PROJECTION = osr.GetUserInputAsWKT(str('epsg:28992'))


def rasterize(points):
    """ Create array. """
    xmin, ymin = points[:, :2].min(0)
    xmax, ymax = points[:, :2].max(0)

    p = math.floor(xmin / WIDTH) * WIDTH
    q = math.floor(ymax / HEIGHT) * HEIGHT

    geo_transform = p, WIDTH, 0, q, 0, -HEIGHT

    indices = np.empty((len(points), 3), 'u4')
    indices[:, 2] = (points[:, 0] - p) / WIDTH
    indices[:, 1] = (q - points[:, 1]) / HEIGHT

    order = indices.view('u4,u4,u4').argsort(order=['f1', 'f2'], axis=0)[:, 0]
    indices = indices[order]

    indices[0, 0] = 0
    py, px = indices[0, 1:]
    for i in range(1, len(indices)):
        same1 = indices[i, 1] == indices[i - 1, 1]
        same2 = indices[i, 2] == indices[i - 1, 2]
        if same1 and same2:
            indices[i, 0] = indices[i - 1, 0] + 1
        else:
            indices[i, 0] = 0

    array = np.full(indices.max(0) + 1, NO_DATA_VALUE)
    array[tuple(indices.transpose())] = points[:, 2][order]
    array = np.ma.masked_values(array, NO_DATA_VALUE)

    return {'array': array,
            'projection': PROJECTION,
            'no_data_value': NO_DATA_VALUE,
            'geo_transform': geo_transform}


def txt2tif(source_path):
    root, ext = os.path.splitext(source_path)
    points = np.loadtxt(source_path)
    kwargs = rasterize(points)
    array = kwargs.pop('array')
    for statistic in 'min', 'max':
        func = getattr(np.ma, statistic)
        kwargs['array'] = func(array, 0).filled(NO_DATA_VALUE)[np.newaxis]
        target_path = root + '_' + statistic + '.tif'
        with datasets.Dataset(**kwargs) as dataset:
            DRIVER.CreateCopy(target_path, dataset, options=OPTIONS)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_path', metavar='FILE')
    return parser


def main():
    """ Call txt2tif with args from parser. """
    return txt2tif(**vars(get_parser().parse_args()))


if __name__ == '__main__':
    exit(main())
