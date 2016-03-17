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

from scipy import ndimage
import numpy as np

from raster_tools import datasets
from raster_tools import utils

from raster_tools import gdal
from raster_tools import gdal_array

GTIF = gdal.GetDriverByName(str('gtiff'))
DTYPE = np.dtype('i8, i8')

COURSES = np.array([(64, 128, 1),
                    (32,   0, 2),
                    (16,   8, 4)], 'u1')

INDICES = COURSES.nonzero()
NUMBERS = COURSES[INDICES][np.newaxis, ...]
OFFSETS = np.array(INDICES).transpose()[np.newaxis] - 1
WEIGHTS = 1 / np.sqrt(np.square(OFFSETS).sum(2))
VECTORS = OFFSETS * WEIGHTS[..., np.newaxis]
INVERSE = COURSES[tuple(-np.array(OFFSETS[0].T) + 1)][np.newaxis]


def get_neighbours(indices):
    """ Return indices to neighbour points of the indices points. """
    array1 = np.array(indices).transpose().reshape(-1, 1, 2)
    array8 = array1 + OFFSETS
    return tuple(array8.reshape(-1, 2).transpose())


def get_look_up_table():
    """ Create and return look-up-table. """
    # resultant vectors
    encode = np.arange(256, dtype='u1')[:, np.newaxis]    # which courses
    select = np.bool8(encode & NUMBERS)[..., np.newaxis]  # which numbers
    result = (select * VECTORS).sum(1)[:, np.newaxis, :]  # what resultant

    # select courses with the highest dotproduct and
    common = (result * VECTORS).sum(2)                    # best direction
    fitted = (common * select[..., 0]).argmax(1)          # fitting encoded
    mapped = NUMBERS[0, fitted]                           # mapping
    mapped[0] = 0
    return mapped


def get_traveled(indices, courses, unique):
    """ Return indices when travelling along courses. """
    # turn indices into points array
    points = np.array(indices).transpose()[:, np.newaxis, :]  # make points

    # determine uphill directions and apply offsets
    encode = courses[indices][:, np.newaxis]                   # which codes
    select = np.bool8(encode & NUMBERS)[..., np.newaxis]       # which courses
    target = (points + select * OFFSETS).reshape(-1, 2)        # apply offsets

    if unique:
        target = np.unique(
            np.ascontiguousarray(target).view(DTYPE)
        ).view(target.dtype).reshape(-1, 2)

    return tuple(target.transpose())                           # return tuple


def calculate_flow_direction(values):
    """
    Single neighbour: Encode directly
    Multiple neighbours:
    - Zero drop: Resolve later, iteratively
    - Nonzero drop: Resolve immediately using look-up table
    """
    # output
    direction = np.zeros_like(values, dtype='u1')

    # calculation of drop per neighbour cell
    factor = np.zeros((3, 3))
    factor[INDICES] = WEIGHTS[0]

    best_drop = np.zeros_like(values)
    for i, j in zip(*factor.nonzero()):
        kernel = np.zeros((3, 3))
        kernel[i, j] = -factor[i, j]
        kernel[1, 1] = +factor[i, j]

        this_drop = ndimage.correlate(values, kernel)

        # same drops add to the direction
        same_drop = (this_drop == best_drop)
        direction[same_drop] += COURSES[i, j]

        # better drops replace the direction
        more_drop = this_drop > best_drop
        direction[more_drop] = COURSES[i, j]
        best_drop[more_drop] = this_drop[more_drop]

    # use look-up-table to eliminate multi-directions for nonzero drops:
    lut = get_look_up_table()
    some_drop = (best_drop > 0)
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
    kwargs = {'structure': np.ones((3, 3))}

    while True:
        undefined = np.bool8(np.log2(direction) % 1)
        edges = undefined - ndimage.binary_erosion(undefined, **kwargs)

        t_index1 = edges.nonzero()
        direction1 = direction[t_index1][:, np.newaxis]

        # find neighbour values
        t_index8 = get_neighbours(t_index1)
        direction8 = direction[t_index8].reshape(-1, 8)

        # neighbour must be in encoded direction
        b_index8a = np.bool8(direction1 & NUMBERS)
        # neighbour must have a defined flow direction
        b_index8b = ~np.bool8(np.log2(direction8) % 1)
        # that direction must not point towards the cell to be defined
        b_index8c = direction8 != INVERSE
        # combined index
        b_index8 = np.logical_and.reduce([b_index8a, b_index8b, b_index8c])

        if not b_index8.any():
            break

        argmax = np.argmax(b_index8, axis=1)
        nonzero = b_index8.any(axis=1)
        superindex = tuple([t_index1[0][nonzero], t_index1[1][nonzero]])
        direction[superindex] = NUMBERS[0, argmax[nonzero]]

    # set still undefined directions (complex depressions) to zero
    return np.where(np.log2(direction) % 1, 0, direction)


class DirectionCalculator(object):
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

    def calculate(self, index_feature):
        # target path
        name = index_feature[str('bladnr')]
        path = os.path.join(self.output_path,
                            name[:3],
                            '{}.tif'.format(name))
        if os.path.exists(path):
            return

        # create directory
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass  # no problem

        geometry = index_feature.geometry().Buffer(0)
        geo_transform = self.geo_transform.shifted(geometry)

        # data
        window = self.geo_transform.get_window(geometry)
        values = self.raster_dataset.ReadAsArray(**window)

        # processing
        direction = calculate_flow_direction(values)[np.newaxis]

        # saving
        options = ['compress=deflate', 'tiled=yes']
        kwargs = {'projection': self.projection,
                  'geo_transform': geo_transform,
                  'no_data_value': None}

        with datasets.Dataset(direction, **kwargs) as dataset:
            GTIF.CreateCopy(path, dataset, options=options)


def flow_dir(index_path, part, **kwargs):
    """
    """
    # select some or all polygons
    index = utils.PartialDataSource(index_path)
    if part is not None:
        index = index.select(part)

    calculator = DirectionCalculator(**kwargs)

    for feature in index:
        calculator.calculate(feature)
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
    flow_dir(**kwargs)
