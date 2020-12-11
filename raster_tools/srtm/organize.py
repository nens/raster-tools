# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-
""" Organize srtm files from stdin. """

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import hashlib
import logging
import os
import sys

logger = logging.getLogger(__name__)


def organize(target_dir, double_dir):

    # init
    checksums = set([])

    for line in sys.stdin:
        source_path = line.strip()
        name = os.path.basename(source_path)
        stash_path = os.path.join(double_dir, source_path)
        target_path = os.path.join(
            target_dir, name[5:8].lower(), name,
        )
        checksum = hashlib.md5(open(source_path).read()).hexdigest()
        # double
        if checksum in checksums or os.path.exists(target_path):
            try:
                os.makedirs(os.path.dirname(stash_path))
            except OSError:
                pass
            os.rename(source_path, stash_path)
            logger.debug('Stashed "{}" [{}]'.format(name, checksum))
            continue
        # not double
        checksums.add(checksum)
        try:
            os.makedirs(os.path.dirname(target_path))
        except OSError:
            pass
        os.rename(source_path, target_path)
        logger.debug('Stored "{}" [{}]'.format(name, checksum))


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target_dir', metavar='TARGET')
    parser.add_argument('double_dir', metavar='DOUBLE')
    return parser


def main():
    """ Call command with args from parser. """
    kwargs = vars(get_parser().parse_args())

    logging.basicConfig(stream=sys.stderr,
                        level=logging.DEBUG,
                        format='%(message)s')

    try:
        organize(**kwargs)
        return 0
    except Exception:
        logger.exception('An exception has occurred.')
        return 1


if __name__ == '__main__':
    exit(main())
