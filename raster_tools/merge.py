# -*- coding: utf-8 -*-
"""
Merge a number of rasterfiles. When any rasterfile is missing, the command does
not proceed, except in the case of rasterfiles with an offset.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import os

from os.path import dirname, exists

import numpy as np

from raster_tools import datasets
from raster_tools import gdal


DRIVER_GDAL_GTIFF = gdal.GetDriverByName(str('gtiff'))
OPTIONS = ['compress=deflate', 'tiled=yes']


def path(text):
    parts = text.split(':')
    path = parts[0]
    try:
        return path, float(parts[1])
    except IndexError:
        return path, None


def merge(source_paths, target_path):
    """ Rebase source on base and write it to target. """
    # skip existing
    if exists(target_path):
        print('{} skipped.'.format(target_path))
        return

    # prepare dirs
    try:
        os.makedirs(dirname(target_path))
    except OSError:
        pass

    for i, (source_path, offset) in enumerate(source_paths):
        # skip when missing sources
        if not exists(source_path):
            print('{} not found.'.format(source_path))
            if offset is not None:
                # except when it has an offset (buildings!)
                return

        # read source dataset
        source_dataset = gdal.Open(str(source_path))
        source_band = source_dataset.GetRasterBand(1)
        source_no_data_value = source_band.GetNoDataValue()
        source_array = source_band.ReadAsArray()

        if i == 0:
            # prepare target array
            target_projection = source_dataset.GetProjection()
            target_geo_transform = source_dataset.GetGeoTransform()
            target_no_data_value = np.finfo(source_array.dtype).max
            target_array = np.full_like(source_array, target_no_data_value)

        # determine mask
        source_mask = (source_array != source_no_data_value)

        # apply offset
        if offset is not None:
            source_array[source_mask] += offset

        # paste
        target_array[source_mask] = source_array[source_mask]

    # write
    kwargs = {
        'projection': target_projection,
        'geo_transform': target_geo_transform,
        'no_data_value': target_no_data_value.item(),
    }

    # write
    with datasets.Dataset(target_array[np.newaxis, ...], **kwargs) as dataset:
        DRIVER_GDAL_GTIFF.CreateCopy(target_path, dataset, options=OPTIONS)
    print('{} created.'.format(target_path))


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)

    # positional arguments
    parser.add_argument(
        'source_paths',
        metavar='SOURCE',
        nargs='+',
        type=path,
        help='Can be path or path:offset',
    )
    parser.add_argument(
        'target_path',
        metavar='TARGET',
        help='Path to target file. Directories will be created if necessary.',
    )

    return parser


def main():
    """ Call command with args from parser. """
    merge(**vars(get_parser().parse_args()))
