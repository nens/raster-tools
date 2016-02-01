# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.
"""
Rasterize zonal statstics (currently percentile or median) into a set
of rasters. The input raster is usually the interpolated dem, to prevent
enclosed geometries having no value.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import os
import sys

from raster_tools import gdal
from raster_tools import ogr
from raster_tools import osr

import numpy as np

from raster_tools import datasets
from raster_tools import postgis
from raster_tools import utils

DRIVER_GDAL_GTIFF = gdal.GetDriverByName(b'gtiff')
DRIVER_GDAL_MEM = gdal.GetDriverByName(b'mem')
DRIVER_OGR_MEM = ogr.GetDriverByName(b'memory')

NO_DATA_VALUE = -3.4028234663852886e+38

logger = logging.getLogger(__name__)


class Rasterizer(object):
    def __init__(self, raster_path, target_dir, table, **kwargs):
        self.postgis_source = postgis.PostgisSource(**kwargs)
        self.target_dir = target_dir
        self.table = table

        self.dataset = gdal.Open(raster_path)
        self.geo_transform = utils.GeoTransform(
            self.dataset.GetGeoTransform()
        )
        self.projection = self.dataset.GetProjection()
        self.sr = osr.SpatialReference(self.projection)
        self.no_data_value = np.finfo('f4').min.item()
        self.kwargs = {'projection': self.projection,
                       'no_data_value': self.no_data_value}

    def path(self, feature):
        leaf = feature[b'bladnr']
        return os.path.join(self.target_dir, leaf[0:3], leaf + '.tif')

    def target(self, feature):
        """ Return empty gdal dataset. """
        geometry = feature.geometry()
        dataset = DRIVER_GDAL_MEM.Create('', 2000, 2500, 1, 6)
        dataset.SetGeoTransform(self.geo_transform.shifted(geometry))
        dataset.SetProjection(self.projection)
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(self.no_data_value)
        band.Fill(self.no_data_value)
        return dataset

    def get_ogr_data_source(self, geometry):
        """ Return geometry wrapped as ogr data source. """
        data_source = DRIVER_OGR_MEM.CreateDataSource('')
        layer = data_source.CreateLayer(b'', self.sr)
        layer_defn = layer.GetLayerDefn()
        feature = ogr.Feature(layer_defn)
        feature.SetGeometry(geometry)
        layer.CreateFeature(feature)
        return data_source

    def single(self, feature, target):
        """
        :param feature: vector feature
        :param target: raster file to write to
        """
        # determine geometry and 1m buffer
        geometry = feature.geometry()
        try:
            geometry_buffer = geometry.Buffer(1).Difference(geometry)
        except RuntimeError:
            # garbage geometry
            return

        # retrieve raster data
        geo_transform = self.geo_transform.shifted(geometry_buffer)
        window = self.geo_transform.get_window(geometry_buffer)
        data = self.dataset.ReadAsArray(**window)
        if data is None:
            return
        data.shape = (1,) + data.shape

        # create ogr data sources with geometry and buffer
        data_source = self.get_ogr_data_source(geometry)
        data_source_buffer = self.get_ogr_data_source(geometry_buffer)

        # determine mask
        mask = np.zeros(data.shape, 'u1')
        dataset_kwargs = {'geo_transform': geo_transform}
        dataset_kwargs.update(self.kwargs)
        with datasets.Dataset(mask, **dataset_kwargs) as dataset:
            gdal.RasterizeLayer(dataset,
                                [1], data_source_buffer[0], burn_values=[1])

        # rasterize the percentile
        try:
            burn = np.percentile(data[mask.nonzero()], 75)
        except IndexError:
            # no data points at all
            return
        gdal.RasterizeLayer(target, [1], data_source[0], burn_values=[burn])

    def rasterize(self, index_feature):
        # prepare or abort
        path = self.path(index_feature)
        if os.path.exists(path):
            return

        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass

        target = self.target(index_feature)

        # fetch geometries from postgis
        data_source = self.postgis_source.get_data_source(
            table=self.table, geometry=index_feature.geometry(),
        )
        # analyze and rasterize
        for bag_feature in data_source[0]:
            self.single(feature=bag_feature, target=target)
        # save
        DRIVER_GDAL_GTIFF.CreateCopy(path,
                                     target,
                                     options=['compress=deflate'])


def command(index_path, part, **kwargs):
    """ Rasterize some postgis tables. """
    index = utils.PartialDataSource(index_path)
    rasterizer = Rasterizer(**kwargs)

    if part is not None:
        index = index.select(part)

    for count, feature in enumerate(index, 1):
        rasterizer.rasterize(feature)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('index_path', metavar='INDEX',
                        help='Path to raster index shapefile')
    parser.add_argument('dbname', metavar='DBNAME',
                        help='Name of the database')
    parser.add_argument('table', metavar='TABLE',
                        help='Table name, including schema (e.g. public.bag)')
    parser.add_argument('raster_path', metavar='RASTER',
                        help='Path to the raster file')
    parser.add_argument('target_dir', metavar='TARGET',
                        help='Target folder for result files')
    parser.add_argument('-s', '--host', default='localhost')
    parser.add_argument('-u', '--user'),
    parser.add_argument('-p', '--password'),
    parser.add_argument('--part',
                        help='Partial processing source, for example "2/3"')
    return parser


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr,
                        level=logging.DEBUG,
                        format='%(message)s')
    try:
        return command(**vars(get_parser().parse_args()))
    except SystemExit:
        raise  # argparse does this
    except:
        logger.exception('An exception has occurred.')

if __name__ == "__main__":
    main()
