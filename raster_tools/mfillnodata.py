# (c) Nelen & Schuurmans, see LICENSE.rst.
# -*- coding: utf-8 -*-
"""
Interpolate nodata regions in a raster using recursive aggregation.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import os

import numpy as np

from raster_tools import datasets
from raster_tools import utils

from raster_tools import fillnodata
from raster_tools import gdal

GTIF = gdal.GetDriverByName(str('gtiff'))


def fill(index_path, source_path, border_path, output_path, part):
    """
    """
    # select some or all polygons
    index = utils.PartialDataSource(index_path)
    if part is not None:
        index = index.select(part)

    filler = fillnodata.Filler(source_path=source_path,
                               border_path=border_path)

    for feature in index:
        # target path
        name = feature[str('name')]
        path = os.path.join(output_path, name[:3], '{}.tif'.format(name))
        if os.path.exists(path):
            continue

        # create directory
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass  # no problem

        # geometries
        inner_geometry = feature.geometry()
        outer_geometry = inner_geometry.Buffer(32, 1)

        # geo transforms
        geo_transform = filler.source.geo_transform
        inner_geo_transform = geo_transform.shifted(inner_geometry)
        outer_geo_transform = geo_transform.shifted(outer_geometry)

        # fill
        result = filler.fill(outer_geometry)

        # cut out
        slices = outer_geo_transform.get_slices(inner_geometry)
        values = result['values'][slices]
        no_data_value = result['no_data_value']
        if np.equal(values, no_data_value).all():
            continue

        # save
        options = ['compress=deflate', 'tiled=yes']
        kwargs = {'projection': filler.source.projection,
                  'geo_transform': inner_geo_transform,
                  'no_data_value': no_data_value.item()}

        with datasets.Dataset(values[np.newaxis], **kwargs) as dataset:
            GTIF.CreateCopy(path, dataset, options=options)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        'index_path',
        metavar='INDEX',
        help='shapefile with geometries and names of output tiles',
    )
    parser.add_argument(
        'source_path',
        metavar='SOURCE',
        help='source GDAL raster dataset with voids'
    )
    parser.add_argument(
        'border_path',
        metavar='BORDER',
        help='Filled aggregated raster that ends the recursive filling.'
    )
    parser.add_argument(
        'output_path',
        metavar='OUTPUT',
        help='target folder',
    )
    parser.add_argument(
        '-p', '--part',
        help='partial processing source, for example "2/3"',
    )
    return parser


def main():
    """ Call fill with args from parser. """
    kwargs = vars(get_parser().parse_args())
    fill(**kwargs)
