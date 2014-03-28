# -*- coding: utf-8 -*-
""" TODO Docstring. """

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import math
import sys

from osgeo import ogr
from osgeo import gdal

from matplotlib import cm
from matplotlib import colors
from PIL import Image
from scipy import ndimage
import numpy as np

ogr.UseExceptions()
gdal.UseExceptions()
logger = logging.getLogger(__name__)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        'marker_path',
        metavar='MARKERS',
    )
    parser.add_argument(
        'elevation_path',
        metavar='ELEVATION',
    )
    parser.add_argument(
        '-s', '--sigma',
        type=float,
        help='For gaussian smoothing.',
    )
    return parser


def rgba(array, cmap='jet'):
    n = colors.Normalize()
    c = cm.get_cmap(cmap)
    return c(n(array), bytes=True)


def rescale(array):
    """ Rescale array to within min, max. """
    return np.interp(
        array, [array.min(), array.max()], [0, 65534],
    ).astype('u2')


def minima(array, sigma):
    """ Return numpy array with marked local minima. """
    if sigma:
        s = ndimage.gaussian_filter(array, sigma)
    else:
        s = array
    m = np.equal(s, ndimage.minimum_filter(s, size=(3, 3)))
    return -1 * m.astype('i2')


def modify(markers, geo_transform, marker_path):
    """ Modify markers in place, returning a conversion array. """
    p, a, b, q, c, d = geo_transform
    (a, b), (c, d) = np.linalg.inv([(a, b), (c, d)])
    datasource = ogr.Open(marker_path)
    layer = datasource[0]
    conversion = np.empty(layer.GetFeatureCount() + 1, 'u8')
    conversion[0] = 0
    I, J = markers.shape
    for k, feature in enumerate(layer, 1):
        # create conversion
        conversion[k] = int(feature[b'kolk_id'])
        # mark appropriate points
        x, y = feature.geometry().GetPoint_2D()
        j = math.floor(int(a * (x - p) + b * (y - q)))
        i = math.floor(int(c * (x - p) + d * (y - q)))
        if 0 <= i < I and 0 <= j < J:
            markers[i, j] = k

    return conversion


def command(marker_path, elevation_path, sigma):
    dataset = gdal.Open(elevation_path)
    logger.debug('read elevation')
    array = rescale(dataset.ReadAsArray())

    logger.debug('secondary markers')
    markers = minima(array=array, sigma=10)

    logger.debug('primary markers')
    conversion = modify(
        markers=markers,
        marker_path=marker_path,
        geo_transform=dataset.GetGeoTransform(),
    )

    logger.debug('watershed')
    watershed = ndimage.watershed_ift(array, markers)
    watershed[watershed == -1] = 0

    #logger.debug('save tif')
    tif = gdal.GetDriverByName(b'gtiff').Create(
        'watershed.tif',
        dataset.RasterXSize,
        dataset.RasterYSize,
        dataset.RasterCount,
        gdal.GDT_Int32,
        ['COMPRESS=DEFLATE'],
    )
    tif.SetProjection(dataset.GetProjection())
    tif.SetGeoTransform(dataset.GetGeoTransform())
    tif.GetRasterBand(1).WriteArray(conversion[watershed])

    logger.debug('save png')
    #edge = np.abs(np.gradient(watershed)).sum(0).astype('b1')
    #e_rgba = rgba(edge, cmap='gray')
    #Image.fromarray(e_rgba).save('edge.png')

    w_rgba = rgba(watershed, cmap='flag')
    w_rgba[markers == -1] = 0, 0, 0, 255
    w_rgba[markers > 0] = 255, 255, 255, 255
    Image.fromarray(w_rgba).save('watershed.png')
    return 0


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return command(**vars(get_parser().parse_args()))
