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

    logger.debug('Copy.')
    try:
        os.makedirs(os.path.dirname(target_path))
    except OSError:
        pass  # it existed
    source = gdal.Open(source_path)
    target = GDAL_DRIVER_GTIFF.CreateCopy(
        target_path,
        source,
        1,
        ['COMPRESS=DEFLATE', 'TILED=YES'],
    )
    target_band = target.GetRasterBand(1)
    logger.debug('Fill.')

    curdir = os.getcwd()
    tmpdir = tempfile.mkdtemp(dir='/dev/shm')
    os.chdir(tmpdir)
    gdal.FillNodata(
        target_band,
        None,
        100,  # search distance
        0,    # smoothing iterations
    )
    os.chdir(curdir)
    os.rmdir(tmpdir)
    logger.info('{} interpolated.'.format(os.path.basename(source_path)))
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
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    return command(**vars(get_parser().parse_args()))
