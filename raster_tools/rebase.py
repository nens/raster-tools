# -*- coding: utf-8 -*-
""" TODO Docstring. """

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import os
import sys

import numpy as np

from raster_tools import gdal
from raster_tools import ogr

from raster_tools import datasets

logger = logging.getLogger(__name__)

DRIVER_GDAL_GTIFF = gdal.GetDriverByName(b'gtiff')


def single(base_path, source_path, target_path, tolerance=None):
    """ Rebase source on base and write it to target. """
    # read datasets
    logger.debug('Read base: {}'.format(base_path))
    base = gdal.Open(base_path)
    base_band = base.GetRasterBand(1)
    base_data = base_band.ReadAsArray()
    base_mask = ~base_band.GetMaskBand().ReadAsArray().astype('b1')

    logger.debug('Read source: {}'.format(source_path))
    source = gdal.Open(source_path)
    source_band = source.GetRasterBand(1)
    source_data = source_band.ReadAsArray()
    source_mask = ~source_band.GetMaskBand().ReadAsArray().astype('b1')

    # calculation
    logger.debug('Determine difference...')
    try:
        no_data_value = np.finfo(base_data.dtype).max
    except ValueError:
        no_data_value = np.iinfo(base_data.dtype).max

    # give all data the same no_data_value
    base_data[base_mask] = no_data_value
    source_data[source_mask] = no_data_value

    # calculate content based on equality or tolerance
    if tolerance is None:
        index = np.not_equal(source_data, base_data)
    else:
        index = np.greater(np.abs(source_data - base_data), tolerance)

    target_data = np.empty_like(source_data)
    target_data.fill(no_data_value)
    target_data[index] = source_data[index]

    # write target
    logger.debug('Write target: {}'.format(target_path))

    target_dir = os.path.dirname(target_path)
    try:
        os.makedirs(target_dir)
    except OSError:
        pass

    kwargs = {
        'projection': base.GetProjection(),
        'no_data_value': no_data_value.item(),
        'geo_transform': base.GetGeoTransform(),
    }

    with datasets.Dataset(target_data[np.newaxis, ...], **kwargs) as dataset:
        DRIVER_GDAL_GTIFF.CreateCopy(
            target_path,
            dataset,
            options=['compress=deflate'],
        )


def rebase(index_path, base_root, source_root, target_root, tolerance):
    """ Rebase files based on features from shapefile. """
    index = ogr.Open(index_path)
    layer = index[0]
    total = layer.GetFeatureCount()

    for count, feature in enumerate(layer, 1):
        leaf = feature[b'bladnr']
        sub = leaf[:3]
        base_path = os.path.join(base_root, sub, leaf + '.tif')
        source_path = os.path.join(source_root, sub, leaf + '.tif')
        target_path = os.path.join(target_root, sub, leaf + '.tif')
        single(tolerance=tolerance,
               base_path=base_path,
               source_path=source_path,
               target_path=target_path)
        gdal.TermProgress_nocb(count / total)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index_path', metavar='INDEX')
    parser.add_argument('base_root', metavar='BASE')
    parser.add_argument('source_root', metavar='SOURCE')
    parser.add_argument('target_root', metavar='TARGET')
    parser.add_argument('-t', '--tolerance', type=float)
    return parser


def main():
    """ Call rebase with args from parser. """
    kwargs = vars(get_parser().parse_args())

    logging.basicConfig(stream=sys.stderr,
                        level=logging.DEBUG,
                        format='%(message)s')

    try:
        rebase(**kwargs)
        return 0
    except:
        logger.exception('An exception has occurred.')
        return 1
