# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.
"""
Calculate unique flow directions for pixels in a digital elevation model.
"""

import argparse
import os

from osgeo import gdal
from scipy import ndimage
import numpy as np

from raster_tools import datasets
from raster_tools import datasources
from raster_tools import groups

GTIF = gdal.GetDriverByName('gtiff')
DTYPE = np.dtype('i8, i8')

COURSES = np.array([(64, 128, 1),
                    (32, 0, 2),
                    (16, 8, 4)], 'u1')

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
    fitted = np.where(
        common.any(1),                                    # any common?
        (common * select[..., 0]).argmax(1),              # select best
        select[..., 0].argmax(1),                         # select any
    )
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

    # assign directions based on zero or positive drops
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

    # use look-up-table to eliminate multi-directions for positive drops:
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
        undefined = ~np.in1d(direction, NUMBERS).reshape(direction.shape)
        edges = undefined ^ ndimage.binary_erosion(undefined, **kwargs)

        t_index1 = edges.nonzero()
        direction1 = direction[t_index1][:, np.newaxis]

        # find neighbour values
        t_index8 = get_neighbours(t_index1)
        direction8 = direction[t_index8].reshape(-1, 8)

        # neighbour must be in encoded direction
        b_index8a = np.bool8(direction1 & NUMBERS)
        # neighbour must have a defined flow direction
        b_index8b = np.in1d(direction8, NUMBERS).reshape(b_index8a.shape)
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
    direction[~np.in1d(direction, NUMBERS).reshape(direction.shape)] = 0
    return direction


class DirectionCalculator(object):
    def __init__(self, output_path, raster_path, cover_path):
        # paths and source data
        self.output_path = output_path
        self.raster_group = groups.Group(gdal.Open(raster_path))
        self.cover_group = groups.Group(gdal.Open(cover_path))

        # geospatial reference
        self.geo_transform = self.raster_group.geo_transform
        self.projection = self.raster_group.projection

    def calculate(self, feature):
        # target path
        name = feature[str('name')]
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

        # geometries
        inner_geometry = feature.geometry()
        outer_geometry = inner_geometry.Buffer(50)

        # geo transforms
        inner_geo_transform = self.geo_transform.shifted(inner_geometry)
        outer_geo_transform = self.geo_transform.shifted(outer_geometry)

        values = self.raster_group.read(outer_geometry)
        cover = self.cover_group.read(outer_geometry)

        # set buildings to maximum dem before calculating directions
        maximum = np.finfo(values.dtype).max
        building = np.logical_and(cover > 1, cover < 15)
        values[building] = maximum

        # processing
        direction = calculate_flow_direction(values)

        # make water undefined
        water = np.zeros_like(cover, dtype='b1')
        water.ravel()[:] = np.in1d(cover, (50, 51, 52, 156, 254))
        direction[water] = 0

        # make buildings undefined
        direction[building] = 0

        # cut out
        slices = outer_geo_transform.get_slices(inner_geometry)
        direction = direction[slices][np.newaxis]

        # saving
        options = ['compress=deflate', 'tiled=yes']
        kwargs = {'no_data_value': 0,
                  'projection': self.projection,
                  'geo_transform': inner_geo_transform}

        with datasets.Dataset(direction, **kwargs) as dataset:
            GTIF.CreateCopy(path, dataset, options=options)


def flow_dir(index_path, part, **kwargs):
    """
    """
    # select some or all polygons
    index = datasources.PartialDataSource(index_path)
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
        help='source GDAL raster dataset with depressions filled',
    )
    parser.add_argument(
        'cover_path',
        metavar='COVER',
        help='functional landuse raster.'
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
