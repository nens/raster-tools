# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
"""
Rasterize a set of queries into a collection of raster tile files.

The queries must be defined in a file. The approach is to infer as little as
possible from the queries. The user is responsible for defining queries that
only return two columns, namely 'geometry' and 'value'. For example:

    select my_geometry as geometry, my_value as value from my_schema.my_table

All statements in the queryfile should be one-liners and NOT ending with ';'.
It is possible, however, to comment lines by starting them with '--'.

The actual queries supplied to the database can be found in a logfile
'rasterize.log'.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from os.path import dirname, exists, join

import argparse
import collections
import getpass
import logging
import os
import re

import psycopg2
import numpy as np

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from raster_tools import datasets

DRIVER_GDAL_GTIFF = gdal.GetDriverByName(str('gtiff'))
DRIVER_GDAL_MEM = gdal.GetDriverByName(str('mem'))
DRIVER_OGR_MEMORY = ogr.GetDriverByName(str('Memory'))

PROJECTION = osr.GetUserInputAsWKT(str('EPSG:28992'))
NO_DATA_VALUE = 255
INITIAL_VALUE = 253
CELLSIZE = 0.5
NAME = str('name')

logging.basicConfig(
    filename='rasterize.log',
    level=logging.INFO,
    format='%(message)s',
)
logger = logging.getLogger(__name__)


Tile = collections.namedtuple(
    'Tile', ['path', 'width', 'height', 'origin', 'polygon'],
)


class Tilesource(object):
    POLYGON = 'POLYGON (({x1} {y1},{x2} {y1},{x2} {y2},{x1} {y2},{x1} {y1}))'

    def __init__(self, tilesourcepath, targetdir):
        self.data_source = ogr.Open(tilesourcepath)
        self.layer = self.data_source[0]
        self.targetdir = targetdir

    def __iter__(self):
        gdal.TermProgress_nocb(0)
        total = self.layer.GetFeatureCount()
        for count, feature in enumerate(self.layer, 1):
            # geometry
            geometry = feature.geometry()
            envelope = geometry.GetEnvelope()
            order = 'x1', 'x2', 'y1', 'y2'
            width = int((envelope[1] - envelope[0]) / CELLSIZE)
            height = int((envelope[3] - envelope[2]) / CELLSIZE)
            origin = envelope[0], envelope[3]
            polygon = self.POLYGON.format(**dict(zip(order, envelope)))

            # path
            name = feature[NAME]
            path = join(self.targetdir, name[0:3], name + '.tif')

            # tile
            tile = Tile(
                path=path,
                width=width,
                height=height,
                origin=origin,
                polygon=polygon,
            )

            yield tile
            gdal.TermProgress_nocb(count / total)


class Raster(object):

    def __init__(self, tile):
        self.path = tile.path
        self.polygon = tile.polygon

        geo_transform = (
            tile.origin[0], CELLSIZE, 0, tile.origin[1], 0, -CELLSIZE,
        )
        self.kwargs = {
            'projection': PROJECTION,
            'geo_transform': geo_transform,
            'no_data_value': NO_DATA_VALUE,
        }

        shape = 1, tile.height, tile.width
        self.data = np.full(shape, INITIAL_VALUE, dtype='u1')
        self.burned = False

    def burn(self, items):
        # source and layer
        data_source = DRIVER_OGR_MEMORY.CreateDataSource('')
        spatial_ref = osr.SpatialReference(PROJECTION)

        # layer definition
        value_name = 'value'
        layer = data_source.CreateLayer(str(''), spatial_ref)
        layer.CreateField(ogr.FieldDefn(str(value_name), ogr.OFTInteger))
        layer_defn = layer.GetLayerDefn()

        # data insertion
        for geometry, value in items:
            feature = ogr.Feature(layer_defn)
            feature[str(value_name)] = value
            feature.SetGeometry(ogr.CreateGeometryFromWkb(bytes(geometry)))
            layer.CreateFeature(feature)

        options = ['ATTRIBUTE={}'.format(value_name)]
        with datasets.Dataset(self.data, **self.kwargs) as dataset:
            gdal.RasterizeLayer(dataset, [1], layer, options=options)
            self.burned = True

    def write(self):
        try:
            os.makedirs(dirname(self.path))
        except OSError:
            pass
        options = ['compress=deflate', 'tiled=yes']
        with datasets.Dataset(self.data, **self.kwargs) as dataset:
            DRIVER_GDAL_GTIFF.CreateCopy(self.path, dataset, options=options)


class Rasterizer(object):
    TEMPLATE = re.sub(' +', ' ', """
        SELECT
            ST_AsBinary(ST_Force2D(ST_CurveToLine(geometry))), value
        FROM
            ({query}) as query
        WHERE
            geometry && ST_GeomFromEWKT('SRID=28992;{polygon}')
    """.replace('\n', ' ')).strip()

    def __init__(self, connection, queryfilepath):
        self.connection = connection
        with open(queryfilepath) as queryfile:
            self.queries = [query.strip()
                            for query in queryfile.readlines()
                            if not query.strip().startswith('--')]

    def rasterize(self, tile):
        logger.info(80 * '-')
        logger.info(tile.path)
        logger.info(80 * '-')
        raster = Raster(tile)
        for query in self.queries:
            # construct sql for this query and this raster
            sql = self.TEMPLATE.format(query=query, polygon=raster.polygon)
            logger.info(sql)

            # query and burn result to raster
            cursor = self.connection.cursor()
            cursor.execute(sql)
            raster.burn(cursor.fetchall())

        # write if appropriate
        if raster.burned:
            raster.write()


def rasterize(tilesourcepath, queryfilepath, targetdir, **kwargs):
    """ """
    connection = psycopg2.connect(**kwargs)
    rasterizer = Rasterizer(connection=connection, queryfilepath=queryfilepath)
    tilesource = Tilesource(tilesourcepath=tilesourcepath, targetdir=targetdir)
    for tile in tilesource:
        if exists(tile.path):
            continue
        rasterizer.rasterize(tile)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('tilesourcepath',
                        metavar='TILESOURCE',
                        help='Path to ogr source for tile layout')
    parser.add_argument('queryfilepath',
                        metavar='QUERYFILE',
                        help='Path to sql file with queries')
    parser.add_argument('targetdir',
                        metavar='TARGETDIR',
                        help='Target directory for raster files')
    parser.add_argument('-d', '--database')
    parser.add_argument('-s', '--host')
    parser.add_argument('-p', '--port')
    parser.add_argument('-u', '--user')
    return parser


def main():
    """ Call command with args from parser. """
    kwargs = vars(get_parser().parse_args())
    kwargs['password'] = getpass.getpass()
    return rasterize(**kwargs)
