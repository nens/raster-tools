# -*- coding: utf-8 -*-
"""
Rebase a raster on some base. If the base does not exist, copy the
source. If the source not exist, skip.
"""

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
OPTIONS = ['compress=deflate', 'tiled=yes']
NAME = '{prefix}{leaf}.tif'


def save(dataset, path):
    """ Save dataset as tif, create directory if needed. """
    logger.debug('Write target: {}'.format(path))
    dirname = os.path.dirname(path)
    try:
        os.makedirs(dirname)
    except OSError:
        pass
    DRIVER_GDAL_GTIFF.CreateCopy(path, dataset, options=OPTIONS)


def rebase(base_path, source_path, target_path, tolerance=None):
    """ Rebase source on base and write it to target. """
    # read datasets
    logger.debug('Read source: {}'.format(source_path))
    try:
        source = gdal.Open(source_path)
    except RuntimeError:
        logger.debug('Error reading source {}, skip.'.format(source_path))
        return
    source_band = source.GetRasterBand(1)
    source_data = source_band.ReadAsArray()
    source_mask = ~source_band.GetMaskBand().ReadAsArray().astype('b1')

    logger.debug('Read base: {}'.format(base_path))
    try:
        base = gdal.Open(base_path)
    except RuntimeError:
        logger.debug('Error reading base {}, copy source.'.format(base_path))
        save(dataset=source, path=target_path)
        return
    base_band = base.GetRasterBand(1)
    base_data = base_band.ReadAsArray()
    base_mask = ~base_band.GetMaskBand().ReadAsArray().astype('b1')

    # calculation
    logger.debug('Determine difference.')
    try:
        no_data_value = np.finfo(source_data.dtype).max
    except ValueError:
        no_data_value = np.iinfo(source_data.dtype).max

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

    # write
    kwargs = {'projection': base.GetProjection(),
              'no_data_value': no_data_value.item(),
              'geo_transform': base.GetGeoTransform()}

    with datasets.Dataset(target_data[np.newaxis, ...], **kwargs) as dataset:
        save(dataset=dataset, path=target_path)


class PathMaker ():
    """ Makes paths. """
    def __init__(self, leaf):
        """ Store common things. """
        self.leaf = leaf

    def make(self, root, prefix=None):
        """ Return a path. """
        if prefix is None:
            prefix = ''
        name = NAME.format(prefix=prefix, leaf=self.leaf)
        return os.path.join(root, self.leaf[:3], name)


def command(index_path, base_root, source_root, target_root, **kwargs):
    """ Rebase files based on features from shapefile. """
    index = ogr.Open(index_path)
    layer = index[0]
    total = layer.GetFeatureCount()

    for count, feature in enumerate(layer, 1):
        path_maker = PathMaker(leaf=feature[b'bladnr'])

        base_path = path_maker.make(root=base_root,
                                    prefix=kwargs.pop('base_prefix'))
        source_path = path_maker.make(root=source_root,
                                      prefix=kwargs.pop('source_prefix'))
        target_path = path_maker.make(root=target_root)

        rebase(base_path=base_path,
               source_path=source_path,
               target_path=target_path, **kwargs)

        gdal.TermProgress_nocb(count / total)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)

    # optional arguments
    parser.add_argument('-b', '--base-prefix')
    parser.add_argument('-s', '--source-prefix')
    parser.add_argument('-t', '--tolerance', type=float)
    parser.add_argument('-v', '--verbose', action='store_true')

    # positional arguments
    parser.add_argument('index_path', metavar='INDEX')
    parser.add_argument('base_root', metavar='BASE')
    parser.add_argument('source_root', metavar='SOURCE')
    parser.add_argument('target_root', metavar='TARGET')
    return parser


def main():
    """ Call command with args from parser. """
    # logging
    kwargs = vars(get_parser().parse_args())
    if kwargs.pop('verbose'):
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level, format='%(message)s')

    # run or fail
    try:
        command(**kwargs)
        return 0
    except:
        logger.exception('An exception has occurred.')
        return 1
