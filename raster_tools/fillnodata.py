# (c) Nelen & Schuurmans, see LICENSE.rst.
# -*- coding: utf-8 -*-
"""
Interpolate nodata regions in a raster using recursive aggregation.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import os

import numpy as np
from scipy import ndimage

from raster_tools import datasets
from raster_tools import utils

from raster_tools import gdal
from raster_tools import gdal_array

GTIF = gdal.GetDriverByName(str('gtiff'))
MARGIN = 2000  # edge effect prevention margin
KERNEL = np.array([[0.0625, 0.1250,  0.0625],
                   [0.1250, 0.2500,  0.1250],
                   [0.0625, 0.1250,  0.0625]])


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
        '-p', '--part',
        help='partial processing source, for example "2/3"',
    )
    return parser


def zoom(values):
    """ Return zoomed array. """
    return values.repeat(2, axis=0).repeat(2, axis=1)


def smooth(values):
    """ Two-step uniform for symmetric smoothing. """
    return ndimage.correlate(values, KERNEL)


def fill(values, no_data_value):
    """
    Fill must return a filled array. It does so by aggregating, requesting
    a fill for that, and zooming back. After zooming back, it smooths
    the filled values and returns.
    """
    mask = values == no_data_value
    if not mask.any():
        # this should end the recursion
        return values

    # aggregate
    filled = fill(**utils.aggregate(values=values,
                                    no_data_value=no_data_value))
    zoomed = zoom(filled)[:values.shape[0], :values.shape[1]]
    return np.where(mask, smooth(zoomed), values)


class Filler(object):
    def __init__(self, output_path, raster_path):
        # paths and source data
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

    def fill(self, index_feature):
        # target path
        name = index_feature[str('name')]
        path = os.path.join(self.output_path,
                            leaf_number[:2],
                            '{}.tif'.format(name))
        if os.path.exists(path):
            return

        # create directory
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass  # no problem

        # geometries
        inner_geometry = index_feature.geometry()
        outer_geometry = inner_geometry.Buffer(0, 1)

        # geo transforms
        inner_geo_transform = self.geo_transform.shifted(inner_geometry)
        outer_geo_transform = self.geo_transform.shifted(outer_geometry)

        # data
        window = self.geo_transform.get_window(outer_geometry)
        values = self.raster_dataset.ReadAsArray(**window)
        no_data_value = self.no_data_value

        if values is None or np.equal(values, no_data_value).all():
            return

        # fill
        filled = fill(values=values, no_data_value=no_data_value)

        # cut out
        slices = outer_geo_transform.get_slices(inner_geometry)
        values = values[slices]
        filled = filled[slices]

        target = np.where(
            np.equal(values, self.no_data_value),
            filled,
            self.no_data_value,
        )[np.newaxis]

        # save
        options = ['compress=deflate', 'tiled=yes']
        kwargs = {'projection': self.projection,
                  'geo_transform': inner_geo_transform,
                  'no_data_value': self.no_data_value.item()}

        with datasets.Dataset(target, **kwargs) as dataset:
            GTIF.CreateCopy(path, dataset, options=options)


def fill(index_path, raster_path, output_path, part):
    """
    """
    # select some or all polygons
    index = utils.PartialDataSource(index_path)
    if part is not None:
        index = index.select(part)

    filler = Filler(raster_path=raster_path, output_path=output_path)

    for feature in index:
        filler.fill(feature)
    return 0


def main():
    """ Call fill with args from parser. """
    kwargs = vars(get_parser().parse_args())
    fill(**kwargs)
