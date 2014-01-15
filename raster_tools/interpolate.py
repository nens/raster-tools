# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument('index_path', metavar='INDEX')
    parser.add_argument('source_path', metavar='SOURCE')
    parser.add_argument('target_path', metavar='TARGET')
    return parser


def command(source_path, target_path, index_path):
    """ Do something spectacular. """
    print(index_path, source_path, target_path)
    return 1


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return command(**vars(get_parser().parse_args()))
