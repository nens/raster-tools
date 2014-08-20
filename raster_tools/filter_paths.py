# -*- coding: utf-8 -*-
""" Filter paths on stdin given some condition. """

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

from osgeo import gdal

logger = logging.getLogger(__name__)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    # add arguments here
    #parser.add_argument(
        #'path',
        #metavar='FILE',
    #)
    return parser


def is_filled(path, factor=0.5):
    """ Return if more than fraction contains data. """
    dataset = gdal.Open(path)
    factors = []
    for i in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(i)
        no_data_value = band.GetNoDataValue()
        data = band.ReadAsArray()
        factors.append((data != no_data_value).sum() / data.size)
    return sum(factors) / len(factors) > factor


def command():
    for line in sys.stdin:
        path = line.strip()
        if is_filled(path):
            sys.stdout.write(line)
    return 0


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return command(**vars(get_parser().parse_args()))
