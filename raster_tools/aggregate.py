# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.
"""
Aggregate by some factor using the median of the aggregated pixels.
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
from raster_tools import gdal_array

from raster_tools import datasets
from raster_tools import utils

GTIF = gdal.GetDriverByName(b'gtiff')

logger = logging.getLogger(__name__)


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
        metavar='RASTER',
        help='source GDAL raster dataset with voids'
    )
    parser.add_argument(
        'output_path',
        metavar='OUTPUT',
        help='target folder',
    )
    parser.add_argument(
        '-f', '--factor',
        metavar='FACTOR',
        default=100,
        help='shrink factor',
    )
    parser.add_argument(
        '-p', '--part',
        help='partial processing source, for example "2/3"',
    )
    return parser


class Aggregator(object):
    def __init__(self, raster_dataset, output_path, factor):
        self.raster_dataset = raster_dataset
        self.output_path = output_path
        self.factor = factor

        self.projection = raster_dataset.GetProjection()
        self.geo_transform = utils.GeoTransform(
            raster_dataset.GetGeoTransform(),
        )

        # no data value
        band = raster_dataset.GetRasterBand(1)
        data_type = band.DataType
        no_data_value = band.GetNoDataValue()
        self.no_data_value = gdal_array.flip_code(data_type)(no_data_value)

    def get_source(self, geometry):
        """ Meta is the three class array. """
        window = self.geo_transform.get_window(geometry)
        return self.raster_dataset.ReadAsArray(**window)

    def execute(self, array):
        f = self.factor
        h, w = array.shape
        h /= f
        w /= f

        view = array.reshape(h, f, w, f).transpose(1, 3, 0, 2)
        copy = view.reshape(f * f, h, w)

        n = self.no_data_value
        return np.median(np.ma.masked_values(copy, n), 0).filled()

    def aggregate(self, index_feature):
        # target path
        leaf_number = index_feature[b'BLADNR']
        path = os.path.join(self.output_path,
                            leaf_number[:3],
                            '{}.tif'.format(leaf_number))
        if os.path.exists(path):
            logger.debug('Target already exists.')
            return

        # create directory
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass  # no problem

        # extract source
        geometry = index_feature.geometry()
        geo_transform1 = self.geo_transform.shifted(geometry)
        geo_transform2 = geo_transform1.scaled(self.factor)

        # aggregate
        source = self.get_source(geometry)
        target = self.execute(source)

        # save
        if np.equal(target, self.no_data_value).all():
            logger.debug('Target contains no data.')
            return

        kwargs = {'projection': self.projection,
                  'geo_transform': geo_transform2,
                  'no_data_value': self.no_data_value.item()}
        with datasets.Dataset(target[np.newaxis], **kwargs) as dataset:
            GTIF.CreateCopy(path, dataset, options=['COMPRESS=DEFLATE'])


def aggregate(index_path, raster_path, output_path, factor, part):
    """
    Aggregate by some factor using the median of the aggregated pixels.
    """
    # select some or all polygons
    index = utils.PartialDataSource(index_path)
    if part is not None:
        index = index.select(part)

    raster_dataset = gdal.Open(raster_path)

    aggregator = Aggregator(factor=factor,
                            output_path=output_path,
                            raster_dataset=raster_dataset)

    for feature in index:
        aggregator.aggregate(feature)
    return 0


def main():
    """ Call aggregate with args from parser. """
    logging.basicConfig(stream=sys.stderr,
                        level=logging.INFO,
                        format='%(message)s')
    try:
        return aggregate(**vars(get_parser().parse_args()))
    except SystemExit:
        raise  # argparse does this
    except:
        logger.exception('An exception has occurred.')
