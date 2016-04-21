# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
"""
Use rasterize module to rasterize landuse from database.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

from raster_tools import rasterize


TABLES = ','.join([
    'utrecht_top10_bermen',
    'utrecht_cbs_gebieden',
    'utrecht_top10_spoor',
    'utrecht_osm_sportvelden',
    'utrecht_top10_bos',
    'utrecht_osm_parkeerterreinen',
    'utrecht_brp_gewassen',
    'utrecht_top10_gras',
    'utrecht_top10_waterlopen_buffer',
    'utrecht_top10_water',
    'utrecht_top10_wegen',
    'utrecht_top10_gebouw_vlak',
    'utrecht_bag_current_ex_functie',
    'utrecht_bag',
    'utrecht_top10_kassen',
])

SCHEMA = 'data_tmp'


COLUMNS = {
    'phy': 'code_physical_landuse',
    'fun': 'code_function',
}


def make_source_path(dbname, host, user, password):
    items = [
        'pg:dbname={dbname}',
        'schemas={schema}',
        'tables={tables}',
    ]
    if host is not None:
        items.append('host={host}')
    if user is not None:
        items.append('user={user}')
    if password is not None:
        items.append('password={password}')
    return ' '.join(items).format(
        dbname=dbname,
        host=host,
        schema=SCHEMA,
        tables=TABLES,
        user=user,
        password=password,
    )


def command(index_path, dbname, host, user, password):
    source_path = make_source_path(dbname=dbname, host=host,
                                   user=user, password=password)
    kwargs = {'index_path': index_path, 'source_path': source_path}
    for key, column in COLUMNS.iteritems():
        target_dir = 'incoming_landuse_{key}'.format(key=key)
        kwargs.update(attribute=column)
        kwargs.update(target_dir=target_dir)
        rasterize.command(**kwargs)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index_path',
                        metavar='INDEX',
                        help='Path to ogr index')
    parser.add_argument('dbname',
                        metavar='DBNAME',
                        help='Path to ogr source')
    parser.add_argument('-s', '--host')
    parser.add_argument('-u', '--user')
    parser.add_argument('-p', '--password')
    return parser


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return command(**vars(get_parser().parse_args()))
