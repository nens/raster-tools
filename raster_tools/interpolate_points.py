# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
"""
Interpolate a points source to a tif, clipped by a polygon source.
"""

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

import numpy as np

from raster_tools import postgis

DRIVER_GDAL_GTIFF = gdal.GetDriverByName(b'gtiff')
DRIVER_GDAL_MEM = gdal.GetDriverByName(b'mem')
DRIVER_OGR_GEOJSON = ogr.GetDriverByName(b'geojson')
DRIVER_OGR_MEMORY = ogr.GetDriverByName(b'memory')

NO_DATA_VALUE = {
    'Real': -3.4028234663852886e+38,
    'Integer': 255,
}

DATA_TYPE = {
    'Real': gdal.GDT_Float32,
    'Integer': gdal.GDT_Byte,
}

POLYGON = 'POLYGON (({x1} {y1},{x2} {y1},{x2} {y2},{x1} {y2},{x1} {y1}))'

logger = logging.getLogger(__name__)
gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()


def get_geometry(data_source):
    """ Return geometry corresponding to the extent of the data source. """
    layer = data_source[0]
    x1, x2, y1, y2 = layer.GetExtent()
    wkt = POLYGON.format(x1=x1, y1=y1, x2=x2, y2=y2)
    return ogr.CreateGeometryFromWkt(wkt)


def get_geotransform(geometry, cellsize):
    """ Return dataset geotransform for an index geometry. """
    a, b, c, d = cellsize[0], 0.0, 0.0, cellsize[1]
    x1, x2, y1, y2 = geometry.GetEnvelope()
    return x1, a, b, y2, c, d


def get_size(geometry, cellsize):
    """ Return dataset size for an index geometry. """
    w, h = cellsize
    x1, x2, y1, y2 = geometry.GetEnvelope()
    return int((x2 - x1) / w), int((y1 - y2) / h)


def get_dataset(geometry, cellsize=(0.5, -0.5)):
    """ Return dataset for an index geometry. """
    width, height = get_size(geometry, cellsize)
    geotransform = get_geotransform(geometry, cellsize)
    data_type = gdal.GDT_Float32
    no_data_value = 3.4028234663852886e+38
    dataset = DRIVER_GDAL_MEM.Create('', width, height, 1, data_type)
    dataset.SetProjection(osr.GetUserInputAsWKT(b'epsg:28992'))
    dataset.SetGeoTransform(geotransform)
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(no_data_value)
    band.Fill(no_data_value)
    return dataset


def get_tmp_data_source(feature, sr):
    """
    Return a memory datasource with feature as its only feature.

    Ogr < 1.10 does not set the layer sr on the feature.
    """
    data_source = DRIVER_OGR_MEMORY.CreateDataSource('')
    layer = data_source.CreateLayer(b'', sr)
    layer.CreateFeature(feature)
    return data_source


def get_coordinates(dataset):
    """ Return x, y arrays. """
    p, a, b, q, c, d = dataset.GetGeoTransform()
    i, j = np.indices((dataset.RasterYSize, dataset.RasterXSize))
    return p + a * j + b * i, q + c * j + d * i


def get_points_and_values(data_source, attribute):
    """ Return points and values arrays. """
    layer = data_source[0]
    items = [(f.geometry().GetPoint_2D(0), f[str(attribute)]) for f in layer]
    points, values = zip(*items)
    return np.array(points), np.array(values)


def interpolate_by_idw(points, values, xi, p=2):
    """ Use idw for interpolation. """
    sum_of_weights = np.zeros(len(xi))
    sum_of_weighted_measurements = np.zeros(len(xi))
    for i in range(values.size):
        distance = np.sqrt(((points[i] - xi) ** 2).sum(1))
        weight = 1.0 / distance ** p
        weighted_measurement = values[i] * weight
        sum_of_weights += weight
        sum_of_weighted_measurements += weighted_measurement
    return sum_of_weighted_measurements / sum_of_weights


def command(index_path, target_dir, buildings, points, where, **kwargs):
    """
    What?
    """
    postgis_source = postgis.PostgisSource(**kwargs)
    index_data_source = ogr.Open(index_path)
    index_layer = index_data_source[0]
    index_total = index_layer.GetFeatureCount()

    # loop leafs
    for index_count, index_feature in enumerate(index_layer, 1):
        leaf = index_feature[b'BLADNR']
        logger.info('Processing {} ({}/{})'.format(
            leaf, index_count, index_total),
        )

        index_geometry = index_feature.geometry()

        # retrieve buildings from database
        logger.debug('Retrieving buildings.')
        building_data_source = postgis_source.get_data_source(
            table=buildings, geometry=index_geometry, where=where,
        )
        #DRIVER_OGR_GEOJSON.CopyDataSource(
            #building_data_source, 'buildings_{}.geojson'.format(leaf),
        #)

        # retrieve points from database
        logger.debug('Retrieving points.')
        building_geometry = get_geometry(building_data_source)
        points_data_source = postgis_source.get_data_source(
            table=points, geometry=building_geometry,
        )
        #DRIVER_OGR_GEOJSON.CopyDataSource(
            #points_data_source, 'points_{}.geojson'.format(leaf),
        #)

        # prepare dataset
        dataset = get_dataset(index_geometry)
        band = dataset.GetRasterBand(1)
        no_data_value = band.GetNoDataValue()
        array = band.ReadAsArray()
        x, y = get_coordinates(dataset)
        building_layer = building_data_source[0]
        building_total = building_layer.GetFeatureCount()

        for building_count, building_feature in enumerate(building_layer, 1):
            # select points in building
            points_data_source[0].SetSpatialFilter(building_feature.geometry())
            if points_data_source[0].GetFeatureCount():

                # rasterize a temporary data source with current building
                tmp_data_source = get_tmp_data_source(
                    feature=building_feature,
                    sr=building_layer.GetSpatialRef(),
                )
                gdal.RasterizeLayer(
                    dataset, [1], tmp_data_source[0], burn_values=[0],
                )
                index = band.ReadAsArray() == 0
                band.Fill(no_data_value)

                # interpolate selected points
                pts, values = get_points_and_values(
                    data_source=points_data_source, attribute='pnt_linear',
                )
                xi = np.array([x[index], y[index]]).transpose()
                array[index] = interpolate_by_idw(pts, values, xi, p=4)

            gdal.TermProgress_nocb(building_count / building_total)

        if (array == no_data_value).all():
            logger.debug('No points found in any building.')
            continue

        # save
        band.WriteArray(array)
        target_path = os.path.join(
            target_dir, leaf[1:4], leaf + '.tif',
        )
        try:
            os.makedirs(os.path.dirname(target_path))
        except OSError:
            pass
        DRIVER_GDAL_GTIFF.CreateCopy(
            target_path, dataset, options=['COMPRESS=DEFLATE'],
        )
        logger.debug('{} Saved.'.format(target_path))


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index_path',
                        metavar='INDEX',
                        help='Path to ogr index')
    parser.add_argument('target_dir',
                        metavar='TARGET',
                        help='Output folder')
    parser.add_argument('buildings',
                        help='my_schema.my_table')
    parser.add_argument('points',
                        help='my_schema.my_table')
    parser.add_argument('-w', '--where',
                        default='AND code_function NOT IN (7, 13)',
                        help='additional where for the buildings table')
    parser.add_argument('-d', '--dbname')
    parser.add_argument('-u', '--user'),
    parser.add_argument('-p', '--password'),
    parser.add_argument('-s', '--host',
                        default='localhost')
    return parser


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return command(**vars(get_parser().parse_args()))
