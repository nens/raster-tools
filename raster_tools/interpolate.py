# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import urllib

from osgeo import gdal

from raster_tools import utils

logger = logging.getLogger(__name__)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument('host_name',
                        metavar='HOST')
    parser.add_argument('index_path',
                        metavar='INDEX',
                        help='OGR Ahn2 index')
    parser.add_argument('target_dir',
                        metavar='TARGET',
                        help='Output folder')
    parser.add_argument('leaf_numbers',
                        metavar='LEAF',
                        nargs='*',
                        help='Ahn2 leaf number')
    return parser


def get_dataset_from_server(dictionary):
    """ Fetch a gdal dataset from a layer on a raster-server. """
    # build the url
    width = str(dictionary['width'])
    url = urllib.urlencode(dict())
    get_parameters = dict(
        polygon=dictionary['polygon'].ExportToWkt(),
        projection=projection,
        width=str(dictionary['width']),
        height=str(dictionary['height']),
        compress=dictionary.get('compress', 'deflate')
    )
    print(url)
    exit()
    url_file = urllib.urlopen(url)
    vsi_file = gdal.VSIFOpenL('myfile', 'w')
    vsi_file.write(url_file.read())
    vsi_file.close()


def command(host_name, index_path, leaf_numbers, target_dir):
    """ Do something spectacular. """
    geo_transforms = utils.get_geo_transforms(index_path)
    for leaf_number in leaf_numbers:
        geo_transform = geo_transforms[leaf_number]
        print(geo_transform)


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return command(**vars(get_parser().parse_args()))
