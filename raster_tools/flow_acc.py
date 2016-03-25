# -*- coding: utf-8 -*-

# (c) Nelen & Schuurmans, see LICENSE.rst.
"""
Accumulate flow.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import os

import numpy as np

from raster_tools import datasets
from raster_tools import groups
from raster_tools import utils

from raster_tools import gdal

GTIF = gdal.GetDriverByName(str('gtiff'))
DTYPE = np.dtype('i8, i8')

COURSES = np.array([(64, 128, 1),
                    (32,   0, 2),
                    (16,   8, 4)], 'u1')

INDICES = COURSES.nonzero()
NUMBERS = COURSES[INDICES][np.newaxis, ...]
OFFSETS = np.array(INDICES).transpose() - 1


def get_traveled(courses):
    """ Return indices when travelling along courses. """
    # turn indices into points array
    height, width = courses.shape
    indices = (np.arange(height).repeat(width),
               np.tile(np.arange(width), height))
    points = np.array(indices).transpose()

    # determine direction and apply offset
    encode = courses[indices][:, np.newaxis]     # which codes
    select = np.bool8(encode & NUMBERS)          # which courses
    target = points + OFFSETS[select.argmax(1)]  # apply offsets

    return tuple(target.transpose())             # return tuple


def accumulate(values):
    """
    Accumulate flow.

    Key principle is the flow array, that relates the source cell A to
    the target cell B as B = flow[A].
    """
    # construct a mapping array for the flow
    size = values.size
    height, width = values.shape
    traveled = get_traveled(values)

    # construct the flow array
    flow = np.empty(size + 1, dtype='i8')
    flow[-1] = size
    flow[:size] = np.where(np.logical_or.reduce([
        values.ravel() == 0,    # undefined cells
        traveled[0] < 0,        # flow-off to the top
        traveled[0] >= height,  # ... bottom
        traveled[1] < 0,        # ... left
        traveled[1] >= width,   # ... right
    ]), size, traveled[0] * width + traveled[1])

    # initial condition
    state = np.arange(size)                  # each cell has a quantity
    flow[flow[flow[state]] == state] = size  # eliminate opposing directions
    accumulation = np.zeros(size, 'u8')      # this contains the result

    # run the flow until nothing changes anymore
    while True:
        state = flow[state]                           # flow the water
        state.sort()                                  # sort
        state = state[:np.searchsorted(state, size)]  # trim
        left = state.size
        if not left:
            break
        accumulation += np.bincount(state, minlength=size)  # count current

    return accumulation.reshape(height, width)


class Accumulator(object):
    def __init__(self, raster_path, output_path):
        # paths and source data
        self.output_path = output_path
        self.raster_group = groups.Group(gdal.Open(raster_path))

        # geospatial reference
        self.geo_transform = self.raster_group.geo_transform
        self.projection = self.raster_group.projection

    def accumulate(self, feature):
        # target path
        name = feature[str('bladnr')]
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
        outer_geometry = inner_geometry.Buffer(200)

        # geo transforms
        inner_geo_transform = self.geo_transform.shifted(inner_geometry)
        outer_geo_transform = self.geo_transform.shifted(outer_geometry)

        # data
        values = self.raster_group.read(outer_geometry)

        # processing
        values = accumulate(values)

        # cut out and convert
        slices = outer_geo_transform.get_slices(inner_geometry)
        values = np.log10(values[slices][np.newaxis] + 1).astype('f4')

        # save
        options = ['compress=deflate', 'tiled=yes']
        kwargs = {'no_data_value': 0,
                  'projection': self.projection,
                  'geo_transform': inner_geo_transform}

        with datasets.Dataset(values, **kwargs) as dataset:
            GTIF.CreateCopy(path, dataset, options=options)


def flow_acc(index_path, part, **kwargs):
    """
    """
    # select some or all polygons
    index = utils.PartialDataSource(index_path)
    if part is not None:
        index = index.select(part)

    accumulator = Accumulator(**kwargs)

    for feature in index:
        accumulator.accumulate(feature)
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
    flow_acc(**kwargs)
