# -*- coding: utf-8 -*-
""" TODO Docstring. """

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import os

from raster_tools import gdal
from raster_tools import ogr
from raster_tools import osr

DRIVER_GDAL_GTIFF = gdal.GetDriverByName(b'gtiff')
DRIVER_GDAL_MEM = gdal.GetDriverByName(b'mem')

PROJECTION = osr.GetUserInputAsWKT(str('epsg:28992'))
NO_DATA_VALUE = 3.4028234663852886e+38
DATA_TYPE = gdal.GDT_Float32

OPTIONS = ['compress=deflate', 'tiled=yes']


def get_geo_transform(geometry, cellsize=(0.5, -0.5)):
    """ Return geotransform. """
    a, b, c, d = cellsize[0], 0.0, 0.0, cellsize[1]
    x1, x2, y1, y2 = geometry.GetEnvelope()
    return x1, a, b, y2, c, d


def constant(index_path, target_dir):
    """ Create tiles with a constant. """
    # properties that are the same for all tiles

    # open index and loop tiles
    index_data_source = ogr.Open(index_path)
    index_layer = index_data_source[0]
    total = index_layer.GetFeatureCount()
    gdal.TermProgress_nocb(0)
    for count, index_feature in enumerate(index_layer, 1):
        leaf_number = index_feature[str('BLADNR')]
        target_path = os.path.join(
            target_dir, leaf_number[:3], leaf_number + '.tif',
        )

        # skip existing
        if os.path.exists(target_path):
            gdal.TermProgress_nocb(count / total)
            continue

        index_geometry = index_feature.geometry()
        geo_transform = get_geo_transform(index_geometry)

        # prepare dataset
        dataset = DRIVER_GDAL_MEM.Create('', 2000, 2500, 1, DATA_TYPE)
        dataset.SetProjection(PROJECTION)
        dataset.SetGeoTransform(geo_transform)

        # raster band
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(NO_DATA_VALUE)
        band.Fill(0)

        try:
            os.makedirs(os.path.dirname(target_path))
        except OSError:
            pass

        DRIVER_GDAL_GTIFF.CreateCopy(target_path, dataset, options=OPTIONS)
        gdal.TermProgress_nocb(count / total)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index_path',
                        metavar='INDEX',
                        help='Path to ogr index')
    parser.add_argument('target_dir',
                        metavar='TARGET',
                        help='Output folder')
    return parser


def main():
    """ Call constant with args from parser. """
    kwargs = vars(get_parser().parse_args())
    constant(**kwargs)
