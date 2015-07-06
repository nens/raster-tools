# -*- coding: utf-8 -*-
""" TODO Docstring. """

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import os
import sys

from raster_tools import gdal
# from raster_tools import gdal_array
from raster_tools import ogr
# from raster_tools import osr

logger = logging.getLogger(__name__)


def single(base_path, source_path, target_path):
    """ Rebase source on base and write it to target. """
    logger.debug(base_path)
    logger.debug(source_path)
    logger.debug(target_path)


def rebase(index_path, base_root, source_root, target_root):
    index = ogr.Open(index_path)
    layer = index[0]
    total = layer.GetFeatureCount()

    for count, feature in enumerate(layer, 1):
        leaf = feature[b'bladnr']
        sub = leaf[:3]
        base_path = os.path.join(base_root, sub, leaf + '.tif')
        source_path = os.path.join(source_root, sub, leaf + '.tif')
        target_path = os.path.join(target_root, sub, leaf + '.tif')
        single(base_path=base_path,
               source_path=source_path,
               target_path=target_path)
        gdal.TermProgress_nocb(count / total)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index_path', metavar='INDEX')
    parser.add_argument('base_root', metavar='BASE')
    parser.add_argument('source_root', metavar='SOURCE')
    parser.add_argument('target_root', metavar='TARGET')
    return parser


def main():
    """ Call rebase with args from parser. """
    kwargs = vars(get_parser().parse_args())

    logging.basicConfig(stream=sys.stderr,
                        level=logging.DEBUG,
                        format='%(message)s')

    try:
        rebase(**kwargs)
        return 0
    except:
        logger.exception('An exception has occurred.')
        return 1
