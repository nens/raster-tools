# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-
"""
Round raster to set decimals.
"""

from os.path import dirname, exists
import argparse
import os
from osgeo import gdal
import numpy as np
from raster_tools import datasets

# output driver and optinos
DRIVER = gdal.GetDriverByName('gtiff')
OPTIONS = ['compress=deflate', 'tiled=yes']

progress = True


class Exchange(object):
    def __init__(self, source_path, target_path):
        """
        Read source, create target array.
        """
        dataset = gdal.Open(source_path)
        band = dataset.GetRasterBand(1)

        self.source = band.ReadAsArray()
        self.no_data_value = band.GetNoDataValue()

        self.shape = self.source.shape

        self.kwargs = {
            'no_data_value': self.no_data_value,
            'projection': dataset.GetProjection(),
            'geo_transform': dataset.GetGeoTransform(),
        }

        self.target_path = target_path
        self.target = np.full_like(self.source, self.no_data_value)

    def round(self, decimals):
        """ Round target. """
        active = self.target != self.no_data_value
        self.target[active] = self.target[active].round(decimals)

    def save(self):
        """ Save. """
        # prepare dirs
        subdir = dirname(self.target_path)
        if subdir:
            os.makedirs(subdir, exist_ok=True)

        # write tiff
        array = self.target[np.newaxis]
        with datasets.Dataset(array, **self.kwargs) as dataset:
            DRIVER.CreateCopy(self.target_path, dataset, options=OPTIONS)


def roundd(source_path, target_path, decimals):
    """ Round decimals. """
    # skip existing
    if exists(target_path):
        print('{} skipped.'.format(target_path))
        return

    # skip when missing sources
    if not exists(source_path):
        print('Raster source "{}" not found.'.format(source_path))
        return

    # read
    exchange = Exchange(source_path, target_path)

    if decimals:
        exchange.round(decimals)

    # save
    exchange.save()


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    # positional arguments
    parser.add_argument(
        'source_path',
        metavar='SOURCE',
    )
    parser.add_argument(
        'target_path',
        metavar='TARGET',
    )
    parser.add_argument(
        '-r', '--round',
        type=int,
        dest='decimals',
        help='Round the result to this number of decimals.',
    )

    return parser


def main():
    """ Call command with args from parser. """
    roundd(**vars(get_parser().parse_args()))
