# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import collections
import itertools
import logging
import multiprocessing
import os
import re
import sys

from osgeo import gdal
from osgeo import gdal_array
from osgeo import ogr
import numpy as np

from raster_tools import utils

logger = logging.getLogger(__name__)

GDAL_DRIVER_GTIFF = gdal.GetDriverByName(b'gtiff')
GDAL_DRIVER_MEM = gdal.GetDriverByName(b'mem')
RE_PATTERN_NAME = re.compile(r'i[0-9][0-9][a-z][nz][12]_[0-9][0-9]')

gdal.UseExceptions()

# silence pyflakes
groups = None
slices = None
shape = None
dtype = None
fillvalue = None

cache = collections.OrderedDict()


def initializer(*initargs):
    """ For multiprocessing. """
    global groups, slices, shape, dtype, fillvalue
    groups = initargs[0]['groups']
    slices = initargs[0]['slices']
    shape = initargs[0]['shape']
    dtype = initargs[0]['dtype']
    fillvalue = initargs[0]['fillvalue']


def func(kwargs):
    """ For multiprocessing. """
    return interpolate(**kwargs)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=(
        'This is the complex interpolation verison taking into account '
        'more leafs than the source leaf only to mitigate edge effects. '
        'Note that other paths in source location may be read.'
    ))
    parser.add_argument('-p', '--processes',
                        default=multiprocessing.cpu_count(),
                        type=int,
                        help='Amount of parallel processes.')
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


def get_groups(index_path):
    """
    Return a dictionary with neighbours by leaf number.

    Loop the index, constructing a dictionary linking center coordinates
    to leaf names. Contstruct another dictionary linking leaf center
    coordinates to group center coordinates.

    From these dicts, construct neighbournames by name. Skip if name
    not in name dict, None if neighbour not in name dict.
    """
    groups = {}

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
        names[tuple(coords[1][1])] = name
        thing[tuple(coords[1][1])] = coords

    # construct result
    for k, v in thing.iteritems():
        name = names.get(k)
        if not name:
            return
        groups[name] = [map(lambda c: names.get(tuple(c)), w) for w in v]

    return groups


def get_properties(source_paths):
    """ Return work array properties based on first source path. """
    # read
    dataset = gdal.Open(source_paths[0])
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data_type = dataset.GetRasterBand(1).DataType

    # prepare
    slices = [[(slice(j * height, (j + 1) * height),
                slice(i * width, (i + 1) * width))
               for i in range(3)] for j in range(3)]
    shape = 3 * height, 3 * width
    dtype = gdal_array.flip_code(data_type)
    fillvalue = np.finfo(dtype).max

    return dict(slices=slices, shape=shape, dtype=dtype, fillvalue=fillvalue)


def fill(array, path):
    """ Return from cache or file. """
    print(len(cache))
    if path in cache:
        array[:] = cache[path]
        return 'from cache'

    try:
        dataset = gdal.Open(path)
    except RuntimeError:
        return 'not exist'

    dataset.ReadAsArray(buf_obj=array)
    mask = np.equal(array, dataset.GetRasterBand(1).GetNoDataValue())
    array[mask] = fillvalue
    cache[path] = array.copy()
    if len(cache) > 10:
        print('pop!')
        cache.popitem(last=False)
    return 'from file'


def interpolate(source_path, target_dir):
    """
    """
    target_path = os.path.join(
        target_dir,
        os.path.splitext(source_path)[0].lstrip(os.path.sep)
    ) + '.tif'
    if os.path.exists(target_path):
        logger.info('{} exists.'.format(os.path.basename(source_path)))
        return 1

    source_name = os.path.splitext(os.path.basename(source_path))[0]
    group = groups[source_name]

    logger.debug('Load.')
    data = np.empty(shape, dtype)
    data.fill(fillvalue)
    for s, n in zip(itertools.chain(*slices), itertools.chain(*group)):
        view = data[s]
        path = RE_PATTERN_NAME.sub(n, source_path)
        print(fill(array=view, path=path))

    logger.debug('Fill.')
    select = slices[1][1]
    dataset = utils.array2dataset(np.expand_dims(data[select], 0))
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(float(fillvalue))
    gdal.FillNodata(
        band,
        None,
        100,  # search distance
        0,    # smoothing iterations
        callback=gdal.TermProgress,
    )
    dataset.FlushCache()

    logger.debug('Save.')
    try:
        os.makedirs(os.path.dirname(target_path))
    except OSError:
        pass  # it existed
    source = gdal.Open(source_path)
    source_band = source.GetRasterBand(1)
    target = GDAL_DRIVER_GTIFF.Create(
        target_path,
        source.RasterXSize,
        source.RasterYSize,
        source.RasterCount,
        source_band.DataType,
        ['COMPRESS=DEFLATE', 'TILED=YES'],
    )
    target.SetGeoTransform(source.GetGeoTransform())
    target.SetProjection(source.GetProjection())
    target_band = target.GetRasterBand(1)
    target_band.WriteArray(data[slices[1][1]])
    target_band.SetNoDataValue(float(fillvalue))
    logger.info('{} interpolated.'.format(os.path.basename(source_path)))
    return 0


def command(index_path, target_dir, source_paths, processes):
    """ Do something spectacular. """
    logger.info('Prepare index.')
    initkwargs = {'groups': get_groups(index_path)}
    initkwargs.update(get_properties(source_paths))
    initargs = [initkwargs]
    iterable = (dict(source_path=source_path,
                     target_dir=target_dir) for source_path in source_paths)

    if processes > 1:
        # multiprocessing
        pool = multiprocessing.Pool(
            processes=processes,
            initializer=initializer,
            initargs=initargs,
        )
        pool.map(func, iterable)
        pool.close()
    else:
        # singleprocessing
        initializer(*initargs)
        map(func, iterable)


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return command(**vars(get_parser().parse_args()))
