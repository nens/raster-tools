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

FLOW_ARRAY = np.array([(64, 128, 1),
                       (32,   0, 2),
                       (16,   8, 4)], 'u1')

FLOW_INDICES = FLOW_ARRAY.nonzero()
FLOW_NUMBERS = FLOW_ARRAY[FLOW_INDICES][np.newaxis, ...]
FLOW_OFFSETS = np.array(FLOW_INDICES).reshape(1, 8, 2) - 1
FLOW_INVERSE = FLOW_ARRAY[tuple(-np.array(FLOW_OFFSETS[0].T) + 1)][np.newaxis]

# FLOW_VECTORS = None  # later, to get rid of trigonometry
# FLOW_WEIGHTS = None  # later, to get rid of explicit weighting


def get_neighbours(indices):
    """ Return indices to neighbour points if the indices points. """
    array1 = np.array(indices).transpose().reshape(-1, 1, 2)
    array8 = array1 + FLOW_OFFSETS
    return tuple(array8.reshape(-1, 2).transpose())


def get_look_up_table():
    """
    Create and return look-up-table.
    """
    # first pass, determine resultant vectors for the encoded components
    encoding = np.arange(0, 256, dtype='u1')
    vector = np.zeros((256, 2))
    for i in 1, 2, 4, 8, 16, 32, 64, 128:
        s = np.bool8(encoding & i)             # select matched encodings
        p = np.pi / 4 * np.log2(i)             # trigonometric phase
        v = np.hstack([np.cos(p), np.sin(p)])  # corresponding vector
        vector[s] += v                         # add vectors

    # second pass, determine the most common component compared to resultant
    common = np.zeros(256)
    mapped = np.zeros_like(encoding)
    for i in 1, 2, 4, 8, 16, 32, 64, 128:
        s = np.bool8(encoding & i)             # select matched encodings
        p = np.pi / 4 * np.log2(i)             # trigonometric phase
        v = np.hstack([np.cos(p), np.sin(p)])  # corresponding vector
        d = (v * vector).sum(1)                # dot product
        x = np.logical_and(s, d >= common)     # common index
        common[x] = d[x]
        mapped[x] = i
    return mapped


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
    """
    Single neighbour: Encode directly
    Multiple neighbours:
    - Zero drop: Resolve later, iteratively
    - Nonzero drop: Resolve immediately using look-up table
    """
    # output and coding
    direction = np.zeros_like(values, dtype='u1')

    # calculation of drop per neighbour cell
    a, b = 1, math.sqrt(2) / 2
    factor = np.array([(b, a, b),
                       (a, 0, a),
                       (b, a, b)])  # TODO replace with weights for use in lut

    best_drop = np.zeros_like(values)
    for i, j in zip(*factor.nonzero()):
        kernel = np.zeros((3, 3))
        kernel[i, j] = -factor[i, j]
        kernel[1, 1] = +factor[i, j]

        this_drop = ndimage.correlate(values, kernel)

        # same drops add to the direction
        same_drop = this_drop == best_drop
        direction[same_drop] += FLOW_ARRAY[i, j]

        # better drops replace the direction
        more_drop = this_drop > best_drop
        direction[more_drop] = FLOW_ARRAY[i, j]
        best_drop[more_drop] = this_drop[more_drop]

    # use look-up-table to eliminate multi-directions for nonzero drops:
    lut = get_look_up_table()
    some_drop = best_drop > 0
    direction[some_drop] = lut[direction[some_drop]]

    # assign outward to edges
    direction[0, -1] = 1
    direction[1:-1, -1] = 2
    direction[-1, -1] = 4
    direction[-1, 1:-1] = 8
    direction[-1, 0] = 16
    direction[1:-1, 0] = 32
    direction[0, 0] = 64
    direction[0, 1:-1] = 128

    # iterate to solve undefined directions where possible
    while True:
        undefined = np.bool8(np.log2(direction) % 1)
        edges = undefined - ndimage.binary_erosion(undefined)

        t_index1 = edges.nonzero()
        direction1 = direction[t_index1]

        # find neigh
        t_index8 = get_neighbours(t_index1)
        direction8 = direction[t_index8].reshape(-1, 8)

        # neighbour must be in encoded direction
        b_index8a = np.bool8(direction1[:, np.newaxis] & FLOW_NUMBERS)
        # neighbour must have a defined flow direction
        b_index8b = ~np.bool8(np.log2(direction8))
        # that direction must not point towards the cell to be defined
        b_index8c = direction1[:, np.newaxis] != FLOW_INVERSE
        # combined index
        b_index8 = np.logical_and.reduce([b_index8a, b_index8b, b_index8c])

        if not b_index8.any():
            break

        argmax = np.argmax(b_index8, axis=1)
        nonzero = b_index8.any(axis=1)
        superindex = tuple([t_index1[0][nonzero], t_index1[1][nonzero]])
        direction[superindex] = FLOW_NUMBERS[0, argmax[nonzero]]

    # set still undefined directions (complex depressions) to zero
    return np.where(np.log2(direction) % 1, 0, direction)


def fill_complex_depressions(direction, values):
    """ Return watershed, the rest remains to be solved later. """
    label, count = ndimage.label(direction == 0)
    hi, lo = values.max(), values.min()
    unsigned = ((values - lo / (hi - lo)) * 65535).astype('u2')
    return ndimage.watershed_ift(unsigned, markers=label)


def calculate_flow_accumulation(direction):
    # start with any defined direction as 0 and the rest as maxint
    # add 1 to any cell pointed to, or first uniq -c it.
    # any cell updated is basis for next step.
    import ipdb
    ipdb.set_trace()


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
        # no_data_value = self.no_data_value

        # processing
        fill_simple_depressions(values)
        direction = calculate_flow_direction(values)
        accumulation = calculate_flow_accumulation(direction)
        values = accumulation

        # save
        values = values[np.newaxis]
        options = ['compress=deflate', 'tiled=yes']
        kwargs = {'projection': self.projection,
                  'geo_transform': geo_transform,
                  # 'no_data_value': no_data_value.item()}
                  'no_data_value': None}

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
