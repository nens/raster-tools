# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
"""
Select some vectors from some database.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import os

from raster_tools import ogr
from raster_tools import postgis
# from raster_tools import osr

DRIVER_OGR_MEMORY = ogr.GetDriverByName(str('Memory'))
DRIVER_OGR_SHAPE = ogr.GetDriverByName(str('ESRI Shapefile'))


class Selector(object):
    def __init__(self, target_path, table, clip, **kwargs):
        self.postgis_source = postgis.PostgisSource(**kwargs)
        self.target_path = target_path
        self.table = table
        self.clip = clip

    def select(self, feature):

        name = feature[str('name')]
        path = os.path.join(self.target_path, name)

        if os.path.exists(path):
            return

        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass

        # fetch geometries from postgis
        data_source = self.postgis_source.get_data_source(
            name=name, table=self.table, geometry=feature.geometry(),
        )

        if self.clip:
            raise NotImplementedError

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_path', metavar='SELECT',
                        help='Path to raster index shapefile')
    parser.add_argument('dbname', metavar='DATABASE',
                        help='Name of the database')
    parser.add_argument('table', metavar='TABLE',
                        help='Table name, including schema (e.g. public.bag)')
    parser.add_argument('target_path', metavar='TARGET',
                        help='Target shapefile')
    parser.add_argument('-u', '--user'),
    parser.add_argument('-p', '--password'),
    parser.add_argument('-s', '--host', default='localhost')
    parser.add_argument('-c', '--clip', action='store_true'),
    return parser


def main():
    """ Call command with args from parser. """
    return command(**vars(get_parser().parse_args()))
