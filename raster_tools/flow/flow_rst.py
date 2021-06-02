# -*- coding: utf-8 -*-

# (c) Nelen & Schuurmans, see LICENSE.rst.
"""
Vectorize flow.
"""

import argparse
import os

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy as np

from raster_tools import datasets
from raster_tools import datasources


GTIF = gdal.GetDriverByName('gtiff')
OPTIONS = ['compress=deflate', 'tiled=yes']


def get_geo_transform(geometry):
    """ Return geotransform. """
    a, b, c, d = 0.5 / 3, 0.0, 0.0, -0.5 / 3
    x1, x2, y1, y2 = geometry.GetEnvelope()
    return x1, a, b, y2, c, d


def rasterize(feature, source_dir, target_dir):
    """ Rasterize streamline shape for a single tile into raster. """
    geo_transform = get_geo_transform(feature.geometry())
    name = feature['name']
    partial_path = os.path.join(name[:3], name)

    # target path
    target_path = os.path.join(target_dir, partial_path) + '.tif'
    if os.path.exists(target_path):
        return

    # create directory
    try:
        os.makedirs(os.path.dirname(target_path))
    except OSError:
        pass  # no problem

    # open source
    source_path = os.path.join(source_dir, partial_path)
    data_source = ogr.Open(source_path)
    layer = data_source[0]

    # create target array
    kwargs = {'no_data_value': 0,
              'geo_transform': geo_transform,
              'array': np.zeros((1, 12500, 10000), 'u1'),
              'projection': osr.GetUserInputAsWKT('epsg:28992')}

    with datasets.Dataset(**kwargs) as dataset:
        for value, attribute in enumerate([2, 3, 4, 4.7], 2):
            layer.SetAttributeFilter('class={}'.format(attribute))
            gdal.RasterizeLayer(dataset, [1], layer, burn_values=[value])

        GTIF.CreateCopy(target_path, dataset, options=OPTIONS)


def flow_rst(index_path, part, **kwargs):
    """
    """
    # select some or all polygons
    index = datasources.PartialDataSource(index_path)
    if part is not None:
        index = index.select(part)

    for feature in index:
        rasterize(feature, **kwargs)
    return 0


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
        'source_dir',
        metavar='FEATURES',
        help='directory with shapefiles',
    )
    parser.add_argument(
        'target_dir',
        metavar='OUTPUT',
        help='target folder',
    )
    parser.add_argument(
        '-p', '--part',
        help='partial processing source, for example "2/3"',
    )
    return parser


def main():
    """ Call aggregate with args from parser. """
    kwargs = vars(get_parser().parse_args())
    flow_rst(**kwargs)
