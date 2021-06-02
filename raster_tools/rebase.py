# -*- coding: utf-8 -*-
"""
Rebase a source raster file on a base raster file, masking cells in the source
raster that are identical to corresponding cells in the base raster. If the
base raster is missing, rebase just copies the source raster.
"""

import argparse
import os
from os.path import dirname, exists

from osgeo import gdal
import numpy as np

from raster_tools import datasets


DRIVER_GDAL_GTIFF = gdal.GetDriverByName('gtiff')
OPTIONS = ['compress=deflate', 'tiled=yes']


def rebase(base_path, source_path, target_path):
    """ Rebase source on base and write it to target. """
    # skip existing
    if exists(target_path):
        print('{} skipped.'.format(target_path))
        return

    # skip when missing sources
    if not exists(source_path):
        print('{} not found.'.format(source_path))

    # prepare dirs
    try:
        os.makedirs(dirname(target_path))
    except OSError:
        pass

    # read source dataset
    source_dataset = gdal.Open(source_path)
    source_band = source_dataset.GetRasterBand(1)
    source_no_data_value = source_band.GetNoDataValue()
    source_array = source_band.ReadAsArray()

    # prepare target array
    target_projection = source_dataset.GetProjection()
    target_geo_transform = source_dataset.GetGeoTransform()
    target_no_data_value = np.finfo(source_array.dtype).max
    target_array = np.full_like(source_array, target_no_data_value)

    # copy active cells
    source_mask = (source_array != source_no_data_value)
    target_array[source_mask] = source_array[source_mask]

    # rebase
    if exists(base_path):
        base_dataset = gdal.Open(base_path)
        base_band = base_dataset.GetRasterBand(1)
        base_no_data_value = base_band.GetNoDataValue()
        base_array = base_band.ReadAsArray()

        # combined mask has active pixels from source and base that are equal
        mask = (base_array != base_no_data_value)
        equal = (source_array == base_array)
        blank = source_mask & mask & equal
        target_array[blank] = target_no_data_value

        method = 'rebase'
    else:
        method = 'copy'

    # write
    kwargs = {
        'projection': target_projection,
        'geo_transform': target_geo_transform,
        'no_data_value': target_no_data_value.item(),
    }

    # write
    with datasets.Dataset(target_array[np.newaxis, ...], **kwargs) as dataset:
        DRIVER_GDAL_GTIFF.CreateCopy(target_path, dataset, options=OPTIONS)
    print('{} created ({}).'.format(target_path, method))


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)

    # positional arguments
    parser.add_argument(
        'base_path',
        metavar='BASE',
        help='Path to base file.',
    )
    parser.add_argument(
        'source_path',
        metavar='SOURCE',
        help='Path to source file.',
    )
    parser.add_argument(
        'target_path',
        metavar='TARGET',
        help='Path to target file. Directories will be created if necessary.',
    )
    return parser


def main():
    """ Call command with args from parser. """
    rebase(**vars(get_parser().parse_args()))
