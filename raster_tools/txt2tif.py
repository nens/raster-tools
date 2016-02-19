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

from scipy import interpolate
import numpy as np

from raster_tools import datasets
from raster_tools import gdal
from raster_tools import osr

WIDTH = 0.5
HEIGHT = 0.5
NO_DATA_VALUE = -9999
DRIVER = gdal.GetDriverByName(str('gtiff'))
OPTIONS = ['compress=deflate', 'tiled=yes']
PROJECTION = osr.GetUserInputAsWKT(str('epsg:28992'))


def rasterize(points):
    """ Create array. """
    xmin, ymin = points[:, :2].min(0)
    xmax, ymax = points[:, :2].max(0)

    p = math.floor(xmin / WIDTH) * WIDTH
    q = math.floor(ymax / HEIGHT) * HEIGHT

    width = int((xmax - p) / WIDTH) + 1
    height = int((q - ymin) / HEIGHT) + 1
    geo_transform = p, WIDTH, 0, q, 0, -HEIGHT

    # simple filling
    array = -9999 * np.ones((1, height, width), dtype='f4')
    index0 = np.uint32((q - points[:, 1]) / HEIGHT)
    index1 = np.uint32((points[:, 0] - p) / WIDTH)
    array[0, index0, index1] = points[:, 2]

    # using interpolation
    # cells = np.indices((height, width)).transpose(1, 2, 0).reshape(-1, 2)
    # rescale = lambda x: (x - (xmin, ymin)) / (xmax - xmin, ymax - ymin)
    # xi = rescale((p, q) + cells[:, ::-1] * (WIDTH, -HEIGHT))
    # pts = rescale(points[:, :2])
    # vals = points[:, 2]
    # array = interpolate.griddata(xi=xi,
                                 # points=pts,
                                 # values=vals,
                                 # fill_value=NO_DATA_VALUE)
    # array.shape = 1, height, width

    print(len(points))
    print(array.size)
    print(len(points) / array.size)

    return {'array': array,
            'projection': PROJECTION,
            'no_data_value': NO_DATA_VALUE,
            'geo_transform': geo_transform}


def command(source_path):
    root, ext = os.path.splitext(source_path)
    points = np.loadtxt(source_path)
    kwargs = rasterize(points)
    target_path = root + '.tif'
    with datasets.Dataset(**kwargs) as dataset:
        DRIVER.CreateCopy(target_path, dataset, options=['compress=deflate'])


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_path', metavar='FILE')
    return parser


def main():
    """ Call command with args from parser. """
    return command(**vars(get_parser().parse_args()))


if __name__ == '__main__':
    exit(main())
