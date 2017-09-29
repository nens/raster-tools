# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
"""
Select some vectors from a posgis database, based on an area of interest
defined in a shape file.

Note that the database geometry must have an SRID defined:

    SELECT UpdateGeometrySRID('ahn2','geom',28992);

And the querying shapefile must have an SRID defined, too:

    gdalsrsinfo epsg:28992 -o wkt > myshape.prj
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import os

from raster_tools import gdal
from raster_tools import ogr
from raster_tools import postgis

DRIVER_OGR_MEMORY = ogr.GetDriverByName(str('Memory'))
DRIVER_OGR_SHAPE = ogr.GetDriverByName(str('ESRI Shapefile'))

COMPATIBLE = {1: (1, 4),
              2: (2, 5),
              3: (3, 6),
              4: (1, 4),
              5: (2, 5),
              6: (3, 6)}


gdal.PushErrorHandler(b'CPLQuietErrorHandler')


class Selector(object):
    def __init__(self, target_path, table, attribute, clip, **kwargs):
        self.postgis_source = postgis.PostgisSource(**kwargs)
        self.target_path = target_path
        self.attribute = attribute
        self.table = table
        self.clip = clip

    def _clip(self, data_source, geometry):
        """ Clip DataSource in-place. """
        l = data_source[0]
        for f in l:
            g = f.geometry()
            c = g.Intersection(geometry)
            if c.GetGeometryType() in COMPATIBLE[g.GetGeometryType()]:
                f.SetGeometryDirectly(c)
                l.SetFeature(f)
            else:
                l.DeleteFeature(f.GetFID())

    def select(self, feature):
        try:
            name = str(feature[str(self.attribute)])
        except ValueError:
            message = 'No attribute "{}" found in selection datasource.'
            print(message.format(self.attribute))
            exit()
        path = os.path.join(self.target_path, name)

        if os.path.exists(path):
            message = '{} Already exists, skipping.'
            print(message.format(path))
            return

        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass

        # fetch geometries from postgis
        geometry = feature.geometry()
        data_source = self.postgis_source.get_data_source(
            name=name, table=self.table, geometry=geometry,
        )

        if self.clip:
            self._clip(data_source=data_source, geometry=geometry)

        DRIVER_OGR_SHAPE.CopyDataSource(data_source, path)


def command(source_path, **kwargs):
    """ Rasterize some postgis tables. """
    selector = Selector(**kwargs)

    source_data_source = ogr.Open(source_path)
    source_layer = source_data_source[0]
    for feature in source_layer:
        selector.select(feature)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('dbname', metavar='DATABASE',
                        help='name of postgis database')
    parser.add_argument('table', metavar='TABLE',
                        help='postgis table [my_schema.]mytable')
    parser.add_argument('source_path', metavar='SELECT',
                        help='path to shapefile defining selection')
    parser.add_argument('target_path', metavar='TARGET',
                        help='where to save the results')
    parser.add_argument('-a', '--attribute', default='name',
                        help='attribute for naming result shapefiles')
    parser.add_argument('-c', '--clip', action='store_true',
                        help='clip geometries that extend outside selection')
    parser.add_argument('-u', '--user'),
    parser.add_argument('-p', '--password'),
    parser.add_argument('-s', '--host', default='localhost')
    return parser


def main():
    """ Call command with args from parser. """
    return command(**vars(get_parser().parse_args()))
