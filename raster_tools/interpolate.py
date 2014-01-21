# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import itertools
import logging
import os
import sys

from osgeo import gdal
from osgeo import gdal_array
from osgeo import ogr
import numpy as np

logger = logging.getLogger(__name__)

# index
groups = {}
slices = None
shape = None
dtype = None
fillvalue = np.finfo(dtype).max


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description="Note that other paths in source location may be read."
    )
    parser.add_argument('index_path',
                        metavar='INDEX',
                        help='OGR Ahn2 index')
    parser.add_argument('target_dir',
                        metavar='TARGET',
                        help='Output folder')
    parser.add_argument('source_paths',
                        metavar='SOURCE',
                        nargs='*',
                        help='Filtered source files')
    return parser


def init_neighbours(index_path):
    """
    Return a dictionary with neighbours by leaf number.

    Loop the index, constructing a dictionary linking center coordinates
    to leaf names. Contstruct another dictionary linking leaf center
    coordinates to group center coordinates.

    From these dicts, construct neighbournames by name. Skip if name
    not in name dict, None if neighbour not in name dict.
    """
    global groups

    offset = np.zeros((3, 3, 2))
    offset[..., 0] = [[-1,  0,  1],
                      [-1,  0,  1],
                      [-1,  0,  1]]
    offset[..., 1] = [[1,  1,  1],
                      [0,  0,  0],
                      [-1, -1, -1]]

    def get_coords(geometry):
        x1, x2, y1, y2 = geometry.GetEnvelope()
        center = (x2 + x1) / 2, (y2 + y1) / 2
        size = x2 - x1, y2 - y1
        return (center + size * offset).tolist()

    names = {}
    thing = {}

    # read the index
    ogr_index_datasource = ogr.Open(index_path)
    ogr_index_layer = ogr_index_datasource[0]
    for ogr_index_feature in ogr_index_layer:
        name = ogr_index_feature[b'BLADNR']
        ogr_index_geometry = ogr_index_feature.geometry()
        coords = get_coords(geometry=ogr_index_geometry)
        names[tuple(coords[0][0])] = name
        thing[tuple(coords[0][0])] = coords

    # construct result
    for k, v in thing.iteritems():
        name = names.get(k)
        if not name:
            return
        groups[name] = [map(lambda c: names.get(tuple(c)), w) for w in v]


def init_geometry(source_paths):
    """ Set global values from first dataset. """
    global slices, shape, dtype, fillvalue
    dataset = gdal.Open(source_paths[0])
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    slices = [[(slice(j * height, (j + 1) * height),
                slice(i * width, (i + 1) * width))
               for i in range(3)] for j in range(3)]
    shape = 3 * height, 3 * width
    dtype = gdal_array.flip_code(dataset.GetRasterBand(1).DataType)
    fillvalue = np.finfo(dtype).max


def interpolate(source_path, target_dir):
    """
    TODO
    - open datasets, or skip if None
    - read into view (fillvalue!)
    - fillnodata
    - write targetpath
    """
    name = os.path.splitext(os.path.basename(source_path))[0]
    group = groups[name]

    # work array
    array = np.array(shape, dtype)
    array.fill(fillvalue)

    from pprint import pprint
    pprint(zip(
        itertools.chain(*slices),
        itertools.chain(*group),
    ))


def command(index_path, target_dir, source_paths):
    """ Do something spectacular. """
    init_geometry(source_paths)
    logger.info('Preparing index')
    init_neighbours(index_path)
    for source_path in source_paths:
        interpolate(source_path, target_dir)


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return command(**vars(get_parser().parse_args()))
