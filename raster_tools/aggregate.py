# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.
"""
Aggregate recursively by taking the mean of quads.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import os

import numpy as np

from raster_tools import datasets
from raster_tools import utils

from raster_tools import gdal
from raster_tools import gdal_array

GTIF = gdal.GetDriverByName(str('gtiff'))


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
        '-i', '--iterations',
        type=int, default=6,
        help='partial processing source, for example "2/3"',
    )
    parser.add_argument(
        '-p', '--part',
        help='partial processing source, for example "2/3"',
    )
    return parser


class Aggregator(object):
    def __init__(self, output_path, raster_path, iterations):
        # paths and source data
        self.iterations = iterations
        self.output_path = output_path
        self.raster_dataset = gdal.Open(raster_path)

        # geospatial reference
        geo_transform = self.raster_dataset.GetGeoTransform()
        self.geo_transform = utils.GeoTransform(geo_transform)
        self.projection = self.raster_dataset.GetProjection()

        # data settings
        band = self.raster_dataset.GetRasterBand(1)
        data_type = band.DataType
        no_data_value = band.GetNoDataValue()
        self.no_data_value = gdal_array.flip_code(data_type)(no_data_value)

    def aggregate(self, index_feature):
        # target path
        name = index_feature[str('name')]
        path = os.path.join(self.output_path,
                            name[:2],
                            '{}.tif'.format(name))
        if os.path.exists(path):
            return

        # create directory
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass  # no problem

        geometry = index_feature.geometry()
        factor = 2 ** self.iterations
        geo_transform = self.geo_transform.shifted(geometry).scaled(factor)

        # data
        window = self.geo_transform.get_window(geometry)
        values = self.raster_dataset.ReadAsArray(**window)
        no_data_value = self.no_data_value

        if values is None:
            return

        # set errors to no data
        index = np.logical_and(
            values != no_data_value,
            np.logical_or(values < -1000, values > 1000),
        )
        values[index] == no_data_value

        if np.equal(values, no_data_value).all():
            return

        # aggregate repeatedly
        kwargs = {'values': values, 'no_data_value': no_data_value}
        for _ in range(self.iterations):
            kwargs = utils.aggregate(**kwargs)

        # save
        values = kwargs['values'][np.newaxis]
        options = ['compress=deflate', 'tiled=yes']
        kwargs = {'projection': self.projection,
                  'geo_transform': geo_transform,
                  'no_data_value': no_data_value.item()}

        with datasets.Dataset(values, **kwargs) as dataset:
            GTIF.CreateCopy(path, dataset, options=options)


def aggregate(index_path, part, **kwargs):
    """
    """
    # select some or all polygons
    index = utils.PartialDataSource(index_path)
    if part is not None:
        index = index.select(part)

    aggregator = Aggregator(**kwargs)

    for feature in index:
        aggregator.aggregate(feature)
    return 0


def main():
    """ Call aggregate with args from parser. """
    kwargs = vars(get_parser().parse_args())
    aggregate(**kwargs)
