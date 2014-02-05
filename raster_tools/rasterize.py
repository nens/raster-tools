# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import os
import sys

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

DRIVER_GDAL_GTIFF = gdal.GetDriverByName(b'gtiff')
DRIVER_GDAL_MEM = gdal.GetDriverByName(b'mem')
DRIVER_OGR_MEMORY = ogr.GetDriverByName(b'Memory')

NO_DATA_VALUE = -3.4028234663852886e+38

logger = logging.getLogger(__name__)
gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()


def get_memory_copy(path):
    """ Return in-memory datasource. """
    logger.debug('Reading {} into memory.'.format(path))
    datasource = ogr.Open(path)
    return DRIVER_OGR_MEMORY.CopyDataSource(datasource, '')


def get_geotransform(geometry, cellsize=(0.5, -0.5)):
    """ Return geotransform. """
    a, b, c, d = cellsize[0], 0.0, 0.0, cellsize[1]
    x1, x2, y1, y2 = geometry.GetEnvelope()
    return x1, a, b, y2, c, d


def command(index_path, source_path, target_dir):
    """ Do something spectacular. """
    # investigate source
    source_datasource = get_memory_copy(source_path)
    source_layer = source_datasource[0]
    source_layer_defn = source_layer.GetLayerDefn()
    if source_layer_defn.GetFieldCount() != 1:
        logger.debug('Source must only have one attribute.')
        exit()
    source_field_defn = source_layer_defn.GetFieldDefn(0)
    source_field_name = source_field_defn.GetName()

    # loop index
    index_datasource = ogr.Open(index_path)
    index_layer = index_datasource[0]
    total = index_layer.GetFeatureCount()
    for count, index_feature in enumerate(index_layer):
        index_geometry = index_feature.geometry()
        source_layer.SetSpatialFilter(index_geometry)
        if not source_layer.GetFeatureCount():
            gdal.TermProgress_nocb(count / total)
            continue

        # prepare dataset
        dataset = DRIVER_GDAL_MEM.Create(
            '', 2000, 2500, 1, gdal.GDT_Float32,
        )
        dataset.SetProjection(osr.GetUserInputAsWKT(b'epsg:28992'))
        dataset.SetGeoTransform(get_geotransform(index_geometry))
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(NO_DATA_VALUE)
        band.Fill(NO_DATA_VALUE)

        # rasterize
        gdal.RasterizeLayer(
            dataset,
            [1],
            source_layer,
            options=['ATTRIBUTE={}'.format(source_field_name)]
        )

        # save
        leaf_number = index_feature[b'BLADNR']
        target_path = os.path.join(
            target_dir, leaf_number[1:4], leaf_number + '.tif',
        )
        try:
            os.makedirs(os.path.dirname(target_path))
        except OSError:
            pass
        DRIVER_GDAL_GTIFF.CreateCopy(
            target_path, dataset, options=['COMPRESS=DEFLATE'],
        )
        gdal.TermProgress_nocb(count / total)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=(
        'Rasterize a vector source into multiple raster files.'
    ))
    parser.add_argument('index_path',
                        metavar='INDEX',
                        help='Path to ogr index')
    parser.add_argument('source_path',
                        metavar='SOURCE',
                        help='Path to ogr source')
    parser.add_argument('target_dir',
                        metavar='TARGET',
                        help='Output folder')
    return parser


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return command(**vars(get_parser().parse_args()))
