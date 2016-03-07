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
import math
import os

from scipy import ndimage
import numpy as np

from raster_tools import datasets
from raster_tools import utils

from raster_tools import gdal
from raster_tools import gdal_array


# create lookup table for combined directions
"""
1. find separate vectors, or complex numbers?
2. pick the one with the greatest dot product with the resultant
"""

encoding = np.arange(0, 256, dtype='u1')
vector = np.zeros((256, 2))
for i in 1, 2, 4, 8, 16, 32, 64, 128:
    s = np.bool8(encoding & i)             # select matched encodings
    p = np.pi / 4 * np.log2(i)             # trigonometric phase
    v = np.hstack([np.cos(p), np.sin(p)])  # corresponding vector
    vector[s] += v                         # add vectors

common = np.zeros((256, 1))
mapped = np.zeros_like(encoding)
for i in 1, 2, 4, 8, 16, 32, 64, 128:
    s = np.bool8(encoding & i)             # select matched encodings
    p = np.pi / 4 * np.log2(i)             # trigonometric phase
    v = np.hstack([np.cos(p), np.sin(p)])  # corresponding vector
    d = (v * vector).sum(1).reshape(-1, 1) # dot product
    x = np.logical_and(s, d >= common)     # common index
    common[x] = d[x]
    mapped[x] = i

print(mapped)



exit()



GTIF = gdal.GetDriverByName(str('gtiff'))


def fill_simple_depressions(values):
    """ Fill simple depressions in-place. """
    footprint = np.array([(1, 1, 1),
                          (1, 0, 1),
                          (1, 1, 1)], dtype='b1')
    edge = ndimage.minimum_filter(values, footprint=footprint)
    locs = edge > values
    values[locs] = edge[locs]


def calculate_flow_direction(values):
    # output and coding
    direction = np.zeros_like(values, dtype='u1')
    code = np.array([(64, 128, 1),
                     (32,   0, 2),
                     (16,   8, 4)], 'u1')

    # calculation of drop per neighbour cell
    a, b = 1, math.sqrt(2) / 2
    factor = np.array([(b, a, b),
                       (a, 0, a),
                       (b, a, b)])

    best_drop = np.zeros_like(values)
    for i, j in zip(*factor.nonzero()):
        kernel = np.zeros((3, 3))
        kernel[i, j] = -factor[i, j]
        kernel[1, 1] = +factor[i, j]

        this_drop = ndimage.correlate(values, kernel)

        # better drops replace the direction
        more_drop = this_drop > best_drop
        direction[more_drop] = code[i, j]
        best_drop[more_drop] = this_drop[more_drop]

        # same drops add to the direction
        same_drop = this_drop == best_drop
        direction[same_drop] += code[i, j]

    import ipdb
    ipdb.set_trace() 

    return direction


class Streamliner(object):
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

    def streamline(self, index_feature):
        # target path
        name = index_feature[str('bladnr')]
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

        geometry = index_feature.geometry().Buffer(-400)
        geo_transform = self.geo_transform.shifted(geometry)

        # data
        window = self.geo_transform.get_window(geometry)
        values = self.raster_dataset.ReadAsArray(**window)
        no_data_value = self.no_data_value

        # processing
        fill_simple_depressions(values)
        values = calculate_flow_direction(values)

        # save
        values = values[np.newaxis]
        options = ['compress=deflate', 'tiled=yes']
        kwargs = {'projection': self.projection,
                  'geo_transform': geo_transform,
                  'no_data_value': no_data_value.item()}

        with datasets.Dataset(values, **kwargs) as dataset:
            GTIF.CreateCopy(path, dataset, options=options)


def streamline(index_path, part, **kwargs):
    """
    """
    # select some or all polygons
    index = utils.PartialDataSource(index_path)
    if part is not None:
        index = index.select(part)

    streamliner = Streamliner(**kwargs)

    for feature in index:
        streamliner.streamline(feature)
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
        metavar='RASTER',
        help='source GDAL raster dataset with voids'
    )
    parser.add_argument(
        'output_path',
        metavar='OUTPUT',
        help='target folder',
    )
    # parser.add_argument(
    #   # '-i', '--iterations',
    #   # type=int, default=6,
    #   # help='partial processing source, for example "2/3"',
    # )
    parser.add_argument(
        '-p', '--part',
        help='partial processing source, for example "2/3"',
    )
    return parser


def main():
    """ Call aggregate with args from parser. """
    kwargs = vars(get_parser().parse_args())
    streamline(**kwargs)
