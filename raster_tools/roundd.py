# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-
"""
Round values, change "no data" value, or both.
"""

from os.path import exists
import argparse
from osgeo.gdal import GetDriverByName, Open
import numpy as np
from raster_tools import datasets

DRIVER = GetDriverByName('gtiff')
OPTIONS = ['compress=deflate', 'tiled=yes']


def roundd(source_path, target_path, decimals=None, no_data_value=None):
    # skip existing
    if exists(target_path):
        print(f'{target_path} skipped.')
        return

    # skip when missing sources
    if not exists(source_path):
        print(f'{target_path} not found.')
        return

    dataset = Open(source_path)
    band = dataset.GetRasterBand(1)

    values = band.ReadAsArray()
    active = values != band.GetNoDataValue()

    kwargs = {
        'projection': dataset.GetProjection(),
        'geo_transform': dataset.GetGeoTransform(),
    }

    # round
    if decimals is not None:
        values[active] = values[active].round(decimals)

    # change "no data" value
    if no_data_value is not None:
        values[~active] = no_data_value
    else:
        no_data_value = band.GetNoDatavalue()

    kwargs["no_data_value"] = no_data_value

    # write tiff
    array = values[np.newaxis]
    with datasets.Dataset(array, **kwargs) as dataset:
        DRIVER.CreateCopy(target_path, dataset, options=OPTIONS)
    print(f'{target_path} written.')


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
    parser.add_argument(
        '-n', '--no-data-value',
        type=float,
        dest='no_data_value',
        help='Use this as new No Data Value.',
    )

    return parser


def main():
    """ Call command with args from parser. """
    roundd(**vars(get_parser().parse_args()))
