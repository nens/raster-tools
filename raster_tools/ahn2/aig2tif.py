# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.
"""
Convert aig to tif. Sources may come from stdin, too.
"""

import argparse
import logging
import multiprocessing
import os
import sys

from osgeo import gdal
from osgeo import osr

from raster_tools import utils

GDAL_GTIFF_DRIVER = gdal.GetDriverByName('gtiff')
GDAL_MEM_DRIVER = gdal.GetDriverByName('mem')

gdal.UseExceptions()
logger = logging.getLogger(__name__)
geo_transforms = None


def initializer(*initargs):
    """ For multiprocessing. """
    global geo_transforms
    geo_transforms = initargs[0]['geo_transforms']


def func(kwargs):
    """ For multiprocessing. """
    return convert(**kwargs)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
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
                        metavar='SOURCES',
                        nargs='*',
                        help='Source raster files')
    return parser


def convert(source_path, target_dir):
    """
    Read, correct, convert and write.
    """
    target_path = os.path.join(
        target_dir,
        os.path.splitext(source_path)[0].lstrip(os.path.sep)
    ) + '.tif'
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
        geo_transforms[os.path.basename(source_path)[1:9]],
    )
    gdal_mem_dataset.SetProjection(
        osr.GetUserInputAsWKT('epsg:28992'),
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


def command(index_path, target_dir, source_paths, processes):
    """ Do something spectacular. """
    logger.info('Prepare index.')
    initargs = [{'geo_transforms': utils.get_geo_transforms(index_path)}]
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
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    return command(**vars(get_parser().parse_args()))
