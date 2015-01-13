# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
"""
Rasterize zonal statstics (currently only median) into a set of rasters.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
# import os
import sys

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from raster_tools import postgis

DRIVER_GDAL_GTIFF = gdal.GetDriverByName(b'gtiff')
DRIVER_GDAL_MEM = gdal.GetDriverByName(b'mem')

NO_DATA_VALUE = -3.4028234663852886e+38

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

logger = logging.getLogger(__name__)


class Rasterizer(object):
    def __init__(self, raster_path, target_dir, table, **kwargs):
        self.postgis_source = postgis.PostgisSource(**kwargs)
        self.dataset = gdal.Open(raster_path)
        self.table = table

    def rasterize(self, feature):
        geometry = feature.geometry()
        data_source = self.postgis_source.get_data_source(table=self.table,
                                                          geometry=geometry)
        layer = data_source[0]
        print(layer.GetFeatureCount())


def command(index_path, **kwargs):
    """ Rasterize some postgis tables. """
    rasterizer = Rasterizer(**kwargs)

    data_source = ogr.Open(index_path)
    layer = data_source[0]
    total = layer.GetFeatureCount()
    for count, feature in enumerate(layer, 1):
        rasterizer.rasterize(feature)
        gdal.TermProgress_nocb(count / total)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index_path', metavar='INDEX')
    parser.add_argument('dbname', metavar='DBNAME')
    parser.add_argument('table', metavar='TABLE')
    parser.add_argument('raster_path', metavar='RASTER')
    parser.add_argument('target_dir', metavar='TARGET')
    parser.add_argument('-u', '--user'),
    parser.add_argument('-p', '--password'),
    parser.add_argument('-s', '--host', default='localhost')
    return parser


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr,
                        level=logging.DEBUG,
                        format='%(message)s')
    try:
        return command(**vars(get_parser().parse_args()))
    except SystemExit:
        raise  # argparse does this
    except:
        logger.exception('An exception has occurred.')
