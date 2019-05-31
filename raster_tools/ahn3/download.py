# -*- coding: utf-8 -*-
"""
Download ahn3 units, using curl.

The INDEX argument should be a shapefile containing the names of the
AHN units (for example 31bz2) in a column named 'name'.
"""

from os.path import join, exists

import argparse
import logging
import os
import shlex
import subprocess
import sys

from osgeo import ogr

logger = logging.getLogger(__name__)


class Router:
    root = 'https://geodata.nationaalgeoregister.nl/ahn3/extract'
    # root = 'http://localhost:8000'
    curl = 'curl --fail --output {path} --retry 3 {url}'

    names = (
        # ('laz', 'ahn3_laz/C_', '.laz'),
        ('dsm', 'ahn3_05m_dsm', 'r_', '.zip'),
        ('dtm', 'ahn3_05m_dtm', 'm_', '.zip'),
    )

    def __init__(self, path):
        self.path = path

        # create dirs
        for name in next(iter(zip(*self.names))):
            try:
                os.makedirs(join(path, name))
            except OSError:
                pass

    def get_directions(self, feature):
        for kind, sub, pre, ext in self.names:
            name = pre + feature[str('name')] + ext
            path = join(self.path, kind, name)
            url = join(self.root, sub, name.upper())
            curl = self.curl.format(url=url, path=path)
            yield path, curl


def download(index_path, target_path):
    router = Router(path=target_path)

    data_source = ogr.Open(index_path)
    layer = data_source[0]
    total = 2 * layer.GetFeatureCount()

    downloaded, processed, notfound, skipped, failed = 0, 0, 0, 0, 0

    for i in range(len(layer)):
        feature = layer[i]
        for path, curl in router.get_directions(feature):
            if exists(path):
                skipped += 1
            else:
                logger.info(curl)
                status = subprocess.call(shlex.split(curl))
                if status == 22:
                    notfound += 1
                elif status:
                    failed += 1
                    if exists(path):
                        os.remove(path)
                else:
                    downloaded += 1
            processed += 1

            # output
            template = ('{progress:.1%}: {processed} processed, '
                        '{downloaded} downloaded, {skipped} skipped, '
                        '{notfound} notfound, {failed} failed.')

            logger.info(template.format(
                progress=processed / total,
                processed=processed,
                downloaded=downloaded,
                skipped=skipped,
                notfound=notfound,
                failed=failed,
            ))


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index_path', metavar='INDEX')
    parser.add_argument('target_path', metavar='TARGET')
    return parser


def main():
    """ Call download with args from parser. """
    kwargs = vars(get_parser().parse_args())

    # logging
    stream = sys.stderr
    level = logging.INFO
    logging.basicConfig(stream=stream, level=level, format='%(message)s')

    # run or fail
    download(**kwargs)
