# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
""" Replace no data value with 0 and pixels with 32767 with 0 too.

Recreates the tifs and leaves old ones as .org files."""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import numpy as np
import os
import sys

import gdal

from raster_tools import datasets

gdal.UseExceptions()

logger = logging.getLogger(__name__)


def fix_nodata():
    for line in sys.stdin:
        # analyze source
        source_path = line.strip()
        source = gdal.Open(source_path)

        array = source.ReadAsArray()[np.newaxis, ...]
        index = np.where(array == -32676)
        no_data_value = source.GetRasterBand(1).GetNoDataValue()

        if no_data_value == 0 and not index[0]:
            logger.debug('Skip {}'.format(source_path))
            continue

        # save modified tif
        logger.debug('Convert {}'.format(source_path))
        array[index] == 0

        kwargs = {'no_data_value': 0,
                  'projection': source.GetProjection(),
                  'geo_transform': source.GetGeoTransform()}
        target_path = '{}.target'.format(source_path)
        driver = source.GetDriver()
        with datasets.Dataset(array, **kwargs) as target:
            target.SetMetadata(source.GetMetadata_List())
            target.GetRasterBand(1).SetUnitType(
                source.GetRasterBand(1).GetUnitType(),
            )
            driver.CreateCopy(target_path,
                              target,
                              options=['compress=deflate'])

        # swap files
        source = None
        backup_path = '{}.bak'.format(source_path)
        os.rename(source_path, backup_path)
        os.rename(target_path, source_path)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    return parser


def main():
    """ Call command with args from parser. """
    kwargs = vars(get_parser().parse_args())

    logging.basicConfig(stream=sys.stderr,
                        level=logging.DEBUG,
                        format='%(message)s')

    try:
        fix_nodata(**kwargs)
        return 0
    except:
        logger.exception('An exception has occurred.')
        return 1
