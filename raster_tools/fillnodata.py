# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import multiprocessing
import os
import sys
import tempfile

from osgeo import gdal
from osgeo import gdal_array
import numpy as np

from raster_tools import utils

GDAL_DRIVER_GTIFF = gdal.GetDriverByName(b'gtiff')

logger = logging.getLogger(__name__)
gdal.UseExceptions()


def func(kwargs):
    """ For multiprocessing. """
    return interpolate(**kwargs)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=(
        'Simply calling the gdal.FillNodata algorithm for the dataset'
    ))
    parser.add_argument('-p', '--processes',
                        default=multiprocessing.cpu_count(),
                        type=int,
                        help='Amount of parallel processes.')
    parser.add_argument('target_dir',
                        metavar='TARGET',
                        help='Output folder')
    parser.add_argument('source_paths',
                        metavar='SOURCE',
                        nargs='*',
                        help='Filtered source files')
    return parser


def interpolate(source_path, target_dir):
    """
    Call gdal interpolation function
    """
    target_path = os.path.join(
        target_dir,
        os.path.splitext(source_path)[0].lstrip(os.path.sep)
    ) + '.tif'
    if os.path.exists(target_path):
        logger.info('{} exists.'.format(os.path.basename(source_path)))
        return 1

    logger.debug('Read.')
    source = gdal.Open(source_path)
    source_band = source.GetRasterBand(1)
    source_data_type = source_band.DataType
    no_data_value = source_band.GetNoDataValue()
    target_array = np.empty(
        (source.RasterCount, source.RasterYSize, source.RasterXSize),
        dtype=gdal_array.flip_code(source_data_type),
    )
    source.ReadAsArray(buf_obj=target_array)
    target = utils.array2dataset(target_array)
    target.SetProjection(source.GetProjection())
    target.SetGeoTransform(source.GetGeoTransform())
    target_band = target.GetRasterBand(1)
    target_band.SetNoDataValue(no_data_value)

    # switch dir
    curdir = os.getcwd()
    tmpdir = tempfile.mkdtemp(dir='/dev/shm')
    os.chdir(tmpdir)

    # fill no data
    iterations = 0
    while no_data_value in target_array:
        logger.debug('Fill')
        mask_array = np.not_equal(target_array, no_data_value).view('u1')
        mask = utils.array2dataset(mask_array)
        mask_band = mask.GetRasterBand(1)
        gdal.FillNodata(
            target_band,
            mask_band,
            100,  # search distance
            0,    # smoothing iterations
        )
        target.FlushCache()
        iterations += 1

    # switch back
    os.chdir(curdir)
    os.rmdir(tmpdir)

    logger.debug('Write.')
    try:
        os.makedirs(os.path.dirname(target_path))
    except OSError:
        pass  # it existed
    GDAL_DRIVER_GTIFF.CreateCopy(
        target_path,
        target,
        1,
        ['COMPRESS=DEFLATE', 'TILED=YES'],
    )
    logger.info('{} interpolated ({}).'.format(
        os.path.basename(source_path), iterations,
    ))
    return 0


def command(target_dir, source_paths, processes):
    """ Do something spectacular. """
    iterable = (dict(source_path=source_path,
                     target_dir=target_dir) for source_path in source_paths)

    if processes > 1:
        # multiprocessing
        pool = multiprocessing.Pool(processes=processes)
        pool.map(func, iterable)
        pool.close()
    else:
        # singleprocessing
        map(func, iterable)


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return command(**vars(get_parser().parse_args()))
