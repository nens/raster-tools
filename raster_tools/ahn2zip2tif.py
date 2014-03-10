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
from osgeo import ogr
from osgeo import osr

from raster_tools import utils

GDAL_GTIFF_DRIVER = gdal.GetDriverByName(b'gtiff')
GDAL_MEM_DRIVER = gdal.GetDriverByName(b'mem')

POLYGON = 'POLYGON (({x1} {y1},{x2} {y1},{x2} {y2},{x1} {y2},{x1} {y1}))'

gdal.UseExceptions()
ogr.UseExceptions()

logger = logging.getLogger(__name__)
index_path = None
character = None


def initializer(*initargs):
    """ For multiprocessing. """
    global index_path, character
    index_path = initargs[0]['index_path']
    character = initargs[0]['character']


def func(kwargs):
    """ For multiprocessing. """
    return convert(**kwargs)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description="Convert aig to tif. Sources may come from stdin, too."
    )
    parser.add_argument('-p', '--processes',
                        default=multiprocessing.cpu_count(),
                        type=int,
                        help='Amount of parallel processes.')
    parser.add_argument('-c', '--character',
                        default='n',
                        help='Character to prepend to filenames.')
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


def get_dataset(path):
    """ Return the vsipath to the first file in a zip. """
    root, ext = os.path.splitext(os.path.basename(path))
    vsipath = '/vsizip/' + os.path.join(path, root)
    return gdal.Open(vsipath)


def get_polygon(dataset):
    """ Return x1, y1, x2, y2 dataset corners. """
    p, a, b, q, c, d = dataset.GetGeoTransform()
    w = dataset.RasterXSize
    h = dataset.RasterYSize
    wkt = POLYGON.format(
        x1=p,
        y1=q + c * w + d * h,
        x2=p + a * w + b * h,
        y2=q,
    )
    return ogr.CreateGeometryFromWkt(wkt)


def create_targets(source):
    """ Return a generator with created targets. """
    polygon = get_polygon(source).Buffer(-1)
    datasource = ogr.Open(index_path)
    layer = datasource[0]
    layer.SetSpatialFilter(polygon)
    wkt = osr.GetUserInputAsWKT(b'epsg:28992')
    no_data_value = source.GetRasterBand(1).GetNoDataValue()
    for feature in layer:
        target = GDAL_MEM_DRIVER.Create('', 2000, 2500, 1, gdal.GDT_Float32)
        target.SetGeoTransform(utils.get_geo_transform(feature))
        target.SetProjection(osr.GetUserInputAsWKT(b'epsg:28992'))
        target.GetRasterBand(1).SetNoDataValue(no_data_value)
        gdal.ReprojectImage(source, target, wkt, wkt, 0, 0.0, 0.125)
        yield feature[b'BLADNR'][1:], target


def convert(source_path, target_dir):
    """
    Read, correct, convert and write.
    """
    logger.info('{} being processed.'.format(os.path.basename(source_path)))
    source = get_dataset(source_path)
    for name, target in create_targets(source):
        target_name = '{}{}.tif'.format(character, name)
        target_path = os.path.join(
            target_dir,
            target_name[1:4],
            target_name,
        )
        try:
            os.makedirs(os.path.dirname(target_path))
        except OSError:
            pass  # it existed
        GDAL_GTIFF_DRIVER.CreateCopy(
            target_path, target, 1, ['COMPRESS=DEFLATE'],
        )
        logger.info('{} converted.'.format(os.path.basename(target_path)))
    return 0


def command(index_path, target_dir, source_paths, processes, character):
    """ Do something spectacular. """
    initargs = [{'index_path': index_path, 'character': character}]
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
