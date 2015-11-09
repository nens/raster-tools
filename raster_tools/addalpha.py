# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.
"""
Add alpha to hillshade map for use with TMS layers.
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
from raster_tools import datasets
from raster_tools import utils
from raster_tools import groups

logger = logging.getLogger(__name__)
driver = gdal.GetDriverByName(str('gtiff'))


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
        band1 = self.group.read(indices)
        alpha = np.interp(band1, (0, 215, 255), (127, 0, 127)).astype('u1')
        combi = np.array([band1, alpha])

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
        with datasets.Dataset(combi, **kwargs) as dataset:
            driver.CreateCopy(path, dataset, options=options)


def addalpha(index_path, raster_path, output_path, part):
    """
    """
    index = utils.PartialDataSource(index_path)
    if part is not None:
        index = index.select(part)

    converter = Converter(raster_path=raster_path, output_path=output_path)

    for feature in index:
        converter.convert(feature)
    return 0


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__
    )
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
    """ Call addalpha with args from parser. """
    # logging
    kwargs = vars(get_parser().parse_args())
    if kwargs.pop('verbose'):
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level, format='%(message)s')

    # run or fail
    try:
        addalpha(**kwargs)
        return 0
    except:
        logger.exception('An exception has occurred.')
        return 1
