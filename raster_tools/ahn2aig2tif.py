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

from osgeo import gdal
from osgeo import osr

from raster_tools import utils

GDAL_GTIFF_DRIVER = gdal.GetDriverByName(b'gtiff')
GDAL_MEM_DRIVER = gdal.GetDriverByName(b'mem')

gdal.UseExceptions()
logger = logging.getLogger(__name__)
geo_transforms = None


def initializer(*initargs):
    """ For multiprocessing. """
    global geo_transforms
    geo_transforms = initargs[0]


def func(kwargs):
    """ For multiprocessing. """
    return convert(**kwargs)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description="Convert aig to tif. Sources may come from stdin, too."
    )
    parser.add_argument('index_path',
                        metavar='INDEX',
                        help='OGR Ahn2 index')
    parser.add_argument('target_dir',
                        metavar='TARGET',
                        help='Output folder')
    parser.add_argument('source_paths',
                        metavar='SOURCES',
                        nargs='*',
                        help='Source raster files')
    return parser


def convert(source_path, target_dir):
    """
    Read, correct, convert and write.
    """
    target_path = os.path.join(target_dir, source_path) + '.tif'
    if os.path.exists(target_path):
        logger.info('{} exists.'.format(os.path.basename(source_path)))
        return 1

    logger.debug('Read.')
    try:
        gdal_source_dataset = gdal.Open(source_path)
        gdal_mem_dataset = GDAL_MEM_DRIVER.CreateCopy('', gdal_source_dataset)
    except RuntimeError:
        logger.warn('{} is broken.'.format(os.path.basename(source_path)))
        return 2

    logger.debug('Set geo transform and projection.')
    gdal_mem_dataset.SetGeoTransform(
        geo_transforms[os.path.basename(source_path)[1:]],
    )
    gdal_mem_dataset.SetProjection(
        osr.GetUserInputAsWKT(b'epsg:28992'),
    )

    logger.debug('Check size')
    width, height = gdal_mem_dataset.RasterXSize, gdal_mem_dataset.RasterYSize
    if (width, height) != (2000, 2500):
        logger.warn('{}x{}:{}'.format(width, height, source_path))

    logger.debug('Write')
    try:
        os.makedirs(os.path.dirname(target_path))
    except OSError:
        pass  # it existed
    GDAL_GTIFF_DRIVER.CreateCopy(
        target_path, gdal_mem_dataset, 1, ['COMPRESS=DEFLATE', 'TILED=YES'],
    )
    logger.info('{} converted.'.format(os.path.basename(source_path)))
    return 0


def command(index_path, target_dir, source_paths):
    """ Do something spectacular. """
    logger.debug('Prepare index.')
    initializer(utils.get_geo_transforms(index_path))

    # single process conversion for sources from arguments
    for source_path in source_paths:
        convert(source_path=source_path, target_dir=target_dir)

    # if there are sources from arguments, ignore stdin
    if source_paths:
        return

    # parallel conversion for sources from stdin
    pool = multiprocessing.Pool(initializer=initializer,
                                initargs=[geo_transforms])
    iterable = (dict(source_path=s.strip(),
                     target_dir=target_dir) for s in sys.stdin)
    pool.map(func, iterable)
    pool.close()


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    return command(**vars(get_parser().parse_args()))
