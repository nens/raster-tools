# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
"""
Rasterize according to some index file from data in an ogr source to
raster files in AHN2 layout. Because of performance problems with the
ogr postgis driver, this module features its own datasource for postgis
connection strings, implemented using psycopg2.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import os
import sys

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from raster_tools import postgis

DRIVER_GDAL_GTIFF = gdal.GetDriverByName(b'gtiff')
DRIVER_GDAL_MEM = gdal.GetDriverByName(b'mem')
DRIVER_OGR_MEMORY = ogr.GetDriverByName(b'Memory')

NO_DATA_VALUE = {
    'Real': -3.4028234663852886e+38,
    'Integer': 255,
}

DATA_TYPE = {
    'Real': gdal.GDT_Float32,
    'Integer': gdal.GDT_Byte,
}

POLYGON = 'POLYGON (({x1} {y1},{x2} {y1},{x2} {y2},{x1} {y2},{x1} {y1}))'

logger = logging.getLogger(__name__)
gdal.UseExceptions()
gdal.PushErrorHandler(b'CPLQuietErrorHandler')
ogr.UseExceptions()
osr.UseExceptions()


def get_geotransform(geometry, cellsize=(0.5, -0.5)):
    """ Return geotransform. """
    a, b, c, d = cellsize[0], 0.0, 0.0, cellsize[1]
    x1, x2, y1, y2 = geometry.GetEnvelope()
    return x1, a, b, y2, c, d


def get_field_name(layer, attribute):
    """
    Return the name of the sole field, or exit if there are more.
    """
    layer_name = layer.GetName()
    layer_defn = layer.GetLayerDefn()
    names = [layer_defn.GetFieldDefn(i).GetName().lower()
             for i in range(layer_defn.GetFieldCount())]
    choices = " or ".join([', '.join(names[:-1])] + names[-1:])
    # attribute given
    if attribute:
        if attribute.lower() in names:
            return attribute.lower()
        print('"{}" not in layer "{}". Choose from {}'.format(
            attribute, layer_name, choices
        ))
    # no attribute given
    else:
        if len(names) == 1:
            return names[0]
        elif not names:
            print('Layer "{}" has no attributes!')
        else:
            print('Layer "{}" has more than one attribute. Use -a option.\n'
                  'Available names: {}'.format(layer_name, choices))
    exit()


def get_ogr_type(data_source, field_names):
    """
    Return the raster datatype corresponding to the field.
    """
    ogr_types = []
    for i, layer in enumerate(data_source):
        layer_defn = layer.GetLayerDefn()
        index = layer_defn.GetFieldIndex(field_names[i])
        field_defn = layer_defn.GetFieldDefn(index)
        ogr_types.append(field_defn.GetTypeName())
    if len(set(ogr_types)) > 1:
        print('Incompatible datatypes:')
        for i, layer in enumerate(data_source):
            print('{:<20} {:<10} {:<7}'.format(
                layer.GetName(), field_names[i], ogr_types[i],
            ))
        exit()
    return ogr_types[0]


def command(index_path, dbname, host, buildings, points, where, target_dir):
    """
    idea is to hit database only 4 times per leaf.
    1. get building geometry column
    2. get all buildings for leaf
    3. get point geometry column
    4. get all points within buildings extent
    index:
        prepare target dataset
        use geometry to query buildings as a layer
        use buildings extent to get points for the leaf
    buildings:
        use rasterize to determine a building mask
        use mask to get all coordinates in buildings
        interpolate points on these coordinates
        write to target tif, done.
    """
    index_data_source = ogr.Open(index_path)
    index_layer = index_data_source[0]
    index_feature = index_layer[0]
    index_geometry = index_feature.geometry()
    index_geometry = index_geometry.Buffer(-450)
    
    postgis_source = postgis.PostgisSource(dbname=dbname, host=host)
    buildings_data_source = postgis_source.get_data_source(
        table=buildings, geometry=index_geometry, where=where,
    )
    print(buildings_data_source)
    


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index_path',
                        metavar='INDEX',
                        help='Path to ogr index')
    parser.add_argument('target_dir',
                        metavar='TARGET',
                        help='Output folder')
    parser.add_argument('buildings',
                        help='my_schema.my_table')
    parser.add_argument('points',
                        help='my_schema.my_table')
    parser.add_argument('-w', '--where',
                        default='AND code_function NOT IN (7, 13)',
                        help='additional where for the buildings table')
    parser.add_argument('-d', '--dbname')
    parser.add_argument('-s', '--host',
                        default='localhost')
    # TODO: user, password
    return parser


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return command(**vars(get_parser().parse_args()))
