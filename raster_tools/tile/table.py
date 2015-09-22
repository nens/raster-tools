# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.
""" Print a table of TMS zoomlevels and corresponding resolutions. """

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import math
import sys

logger = logging.getLogger(__name__)


class Resolution(object):
    def __init__(self, latitude):
        self.width = 40075000 * math.cos(math.radians(latitude))

    def __call__(self, zoomlevel):
        return self.width / (2 ** (zoomlevel + 8))


def zoomtable(latitude):
    """
    wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Resolution_and_Scale
    """
    resolution = Resolution(latitude)

    print('zoom meter / px')
    print('---- ----------')

    template = '{:4d}{:11.3f}'
    for zoomlevel in range(0, 22):
        print(template.format(zoomlevel, resolution(zoomlevel)))


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-l', '--latitude', type=float, default=52.0)
    return parser


def main():
    """ Call zoomtable with args from parser. """
    # logging
    kwargs = vars(get_parser().parse_args())
    if kwargs.pop('verbose'):
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level, format='%(message)s')

    # run or fail
    try:
        zoomtable(**kwargs)
        return 0
    except:
        logger.exception('An exception has occurred.')
        return 1


if __name__ == '__main__':
    exit(main())
