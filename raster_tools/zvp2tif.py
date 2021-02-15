#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create tif rasters from ZVP points by linear interpolation using griddata.
"""

import argparse
import logging
import math
import os
import sys
import zipfile

from osgeo import gdal
from osgeo import osr
from scipy import interpolate
import numpy as np

from raster_tools import datasets

logger = logging.getLogger(__name__)

WIDTH = 0.25
HEIGHT = 0.25
NO_DATA_VALUE = -9999
DRIVER = gdal.GetDriverByName('gtiff')
PROJECTION = osr.GetUserInputAsWKT('epsg:3043')


def read(archive, name):
    """ Read from zip into points. """
    logger.debug('Count lines in "{}".'.format(name))
    total = 100000
    with archive.open(name) as fobj:
        total = fobj.read().count('\n')
    points = np.empty((total, 3), dtype='f4')
    logger.debug('Reading points from "{}".'.format(name))
    with archive.open(name) as fobj:
        for count, line in enumerate(fobj):
            if count == total:
                break
            points[count] = line.split(',')
            gdal.TermProgress_nocb((count + 1) / total)
    return points


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
    # array = -9999 * np.ones((1, height, width), dtype='f4')
    # index0 = np.uint32((q - points[:, 1]) / HEIGHT)
    # index1 = np.uint32((points[:, 0] - p) / WIDTH)
    # array[0, index0, index1] = points[:, 2]

    def rescale(x):
        return (x - (xmin, ymin)) / (xmax - xmin, ymax - ymin)

    # using interpolation
    cells = np.indices((height, width)).transpose(1, 2, 0).reshape(-1, 2)
    xi = rescale((p, q) + cells[:, ::-1] * (WIDTH, -HEIGHT))
    pts = rescale(points[:, :2])
    vals = points[:, 2]
    array = interpolate.griddata(xi=xi,
                                 points=pts,
                                 values=vals,
                                 fill_value=NO_DATA_VALUE)
    array.shape = 1, height, width

    return {'array': array,
            'projection': PROJECTION,
            'no_data_value': NO_DATA_VALUE,
            'geo_transform': geo_transform}


def convert(archive, name):
    points = read(archive=archive, name=name)
    kwargs = rasterize(points)
    path = '{}.tif'.format(os.path.splitext(os.path.basename(name))[0])
    logger.debug('Saving to "{}".'.format(path))
    with datasets.Dataset(**kwargs) as dataset:
        DRIVER.CreateCopy(path, dataset, options=['compress=deflate'])


def is_new(name):
    path = '{}.tif'.format(os.path.splitext(os.path.basename(name))[0])
    return not os.path.exists(path)


def command(path):
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith('txt') and 'FINAL' not in name and is_new(name):
                convert(archive=archive, name=name)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        'path',
        metavar='FILE',
    )
    return parser


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr,
                        level=logging.DEBUG,
                        format='%(message)s')
    try:
        return command(**vars(get_parser().parse_args()))
    except SystemExit:
        raise  # argparse does this
    except Exception:
        logger.exception('An exception has occurred.')


if __name__ == '__main__':
    exit(main())
