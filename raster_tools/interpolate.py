# (c) Nelen & Schuurmans, see LICENSE.rst.
# -*- coding: utf-8 -*-
"""
Interpolate nodata regions in a raster using IDW.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)

import numpy as np
from scipy import ndimage

from raster_tools import datasets
from raster_tools import utils

from raster_tools import gdal
from raster_tools import gdal_array

GTIF = gdal.GetDriverByName(b'gtiff')
MARGIN = 0  # edge effect prevention margin


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
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='be more verbose',
    )
    return parser


def fold(data):
    """ Return folded array. """
    return np.dstack([data[0::2, 0::2],
                      data[0::2, 1::2],
                      data[1::2, 0::2],
                      data[1::2, 1::2]])


def pad_and_agg(data, no_data_value, pad):
    """ Pad, fold, return. """
    s1, s2 = data.shape
    p1, p2 = pad
    result = np.empty(((s1 + p1) / 2, (s2 + p2) / 2), dtype=data.dtype)

    # body
    ma = np.ma.masked_values(
        fold(data[:s1 - p1, :s2 - p2]),
        no_data_value,
    )
    result[:(s1 - p1) / 2, :(s2 - p2) / 2] = ma.mean(2).filled(no_data_value)

    if p1 and p2:
        # corner pixel
        result[-1, -1] = data[-1, -1]
    if p1:
        # bottom row
        ma = np.ma.masked_values(
            fold(data[-1:, :s2 - p2].repeat(2, axis=0)),
            no_data_value,
        )
        result[-1:, :(s2 - p2) / 2] = ma.mean(2).filled(no_data_value)
    if p2:
        # right column
        ma = np.ma.masked_values(fold(
            data[:s1 - p1, -1:].repeat(2, axis=1)),
            no_data_value,
        )
        result[:(s1 - p1) / 2:, -1:] = ma.mean(2).filled(no_data_value)

    return result


def zoom(data):
    """ Return zoomed array. """
    return data.repeat(2, axis=0).repeat(2, axis=1)


def fill(data, no_data_value):
    """
    Fill must return a filled array. It does so by aggregating, requesting a fill for that, and zooming back. After zooming back, it fills and smooths the data and returns.
    """
    if no_data_value not in data:
        return data  # this should be the top level of the challenge at hand

    # determine the structure
    margin = tuple(n % 2 for n in data.shape)

    # get a work array with only small holes remaining to be filled
    above = pad_and_agg(pad=margin, data=data, no_data_value=no_data_value)
    above_filled = fill(data=above, no_data_value=no_data_value)
    
    # get back down, but we have to distuinguish between our agged values and the above filled values, which we should keep.
    zoomed = zoom(above_filled)[:data.shape[0], :data.shape[1]]
    # index agged?  ==> redo pixel-by-pixel
    # index filled? ==> keep



    # fill them!
    above_filled[this == no_data_value] = 7

    return filled

    # return the zoomed work array
    result = np.where(
        np.equal(data, no_data_value),
        ndimage.uniform_filter(filled)
        data,
    )  # or so



class Interpolator(object):
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

    def interpolate(self, index_feature):
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

        # geometries
        inner_geometry = index_feature.geometry()
        outer_geometry = inner_geometry.Buffer(0, 1)

        # geo transforms
        inner_geo_transform = self.geo_transform.shifted(inner_geometry)
        outer_geo_transform = self.geo_transform.shifted(outer_geometry)

        # data
        window = self.geo_transform.get_window(outer_geometry)
        source = self.raster_dataset.ReadAsArray(**window)
        no_data_value = self.no_data_value

        if np.equal(source, no_data_value).all():
            logger.debug('Source contains no data.')
            return

        # fill
        filled = fill(data=source, no_data_value=no_data_value)

        # cut out
        slices = outer_geo_transform.get_slices(inner_geometry)
        source = source[slices]
        filled = filled[slices]

        target = np.where(
            np.equal(source, self.no_data_value),
            filled,
            self.no_data_value,
        )[np.newaxis]

        # save
        kwargs = {'projection': self.projection,
                  'geo_transform': inner_geo_transform,
                  'no_data_value': self.no_data_value.item()}
        with datasets.Dataset(target, **kwargs) as dataset:
            GTIF.CreateCopy(path, dataset, options=['COMPRESS=DEFLATE'])


def interpolate(index_path, raster_path, output_path, part):
    """
    - interpolate all voids per feature at once
    - write to output according to index
    """
    # select some or all polygons
    index = utils.PartialDataSource(index_path)
    if part is not None:
        index = index.select(part)

    interpolator = Interpolator(raster_path=raster_path,
                                output_path=output_path)

    for feature in index:
        interpolator.interpolate(feature)
    return 0


def main():
    """ Call interpolate with args from parser. """
    kwargs = vars(get_parser().parse_args())

    level = logging.DEBUG if kwargs.pop('verbose') else logging.INFO
    logging.basicConfig(**{'level': level,
                           'stream': sys.stderr,
                           'format': '%(message)s'})

    interpolate(**kwargs)
