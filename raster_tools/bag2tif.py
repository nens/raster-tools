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

from os.path import dirname, exists, isdir, join

import argparse
import getpass
import os

from raster_tools import gdal
from raster_tools import ogr
from raster_tools import osr

import numpy as np

from raster_tools import datasets
from raster_tools import datasources
from raster_tools import groups
from raster_tools import postgis

DRIVER_GDAL_GTIFF = gdal.GetDriverByName('gtiff')
DRIVER_GDAL_MEM = gdal.GetDriverByName('mem')
DRIVER_OGR_MEM = ogr.GetDriverByName('memory')

NO_DATA_VALUE = -3.4028234663852886e+38
FLOOR_ATTRIBUTE = 'floor'
CELLSIZE = 0.5


class Rasterizer:
    def __init__(self, table, raster_path, output_path, floor, **kwargs):
        # postgis
        self.postgis_source = postgis.PostgisSource(**kwargs)
        self.table = table

        # raster
        if isdir(raster_path):
            raster_datasets = [gdal.Open(join(raster_path, path))
                               for path in sorted(os.listdir(raster_path))]
        else:
            raster_datasets = [gdal.Open(raster_path)]
        self.raster_group = groups.Group(*raster_datasets)

        # properties
        self.projection = self.raster_group.projection
        self.geo_transform = self.raster_group.geo_transform
        self.no_data_value = self.raster_group.no_data_value.item()

        self.kwargs = {
            'projection': self.projection,
            'no_data_value': self.no_data_value,
        }

        # output
        self.output_path = output_path
        self.floor = floor

    @property
    def sr(self):
        return osr.SpatialReference(self.projection)

    def path(self, feature):
        leaf = feature['name']
        return join(self.output_path, leaf[0:3], leaf + '.tif')

    def target(self, feature):
        """ Return empty gdal dataset. """
        geometry = feature.geometry()
        envelope = geometry.GetEnvelope()
        width = int((envelope[1] - envelope[0]) / CELLSIZE)
        height = int((envelope[3] - envelope[2]) / CELLSIZE)
        dataset = DRIVER_GDAL_MEM.Create('', width, height, 1, 6)
        dataset.SetGeoTransform(self.geo_transform.shifted(geometry))
        dataset.SetProjection(self.projection)
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(self.no_data_value)
        band.Fill(self.no_data_value)
        return dataset

    def get_ogr_data_source(self, geometry):
        """ Return geometry wrapped as ogr data source. """
        data_source = DRIVER_OGR_MEM.CreateDataSource('')
        layer = data_source.CreateLayer('', self.sr)
        layer_defn = layer.GetLayerDefn()
        feature = ogr.Feature(layer_defn)
        feature.SetGeometry(geometry)
        layer.CreateFeature(feature)
        return data_source

    def determine_floor_level(self, feature):
        """
        Return boolean if a floor level was computed and assigned.

        Add assign a computed floor level to the supplied feature.

        :param feature: feature with floor column.
        """
        # determine geometry and 1m buffer
        geometry = feature.geometry()
        try:
            buffer_geometry = geometry.Buffer(1).Difference(geometry)
        except RuntimeError:
            # garbage geometry
            return False

        # read raster data for the extent of the buffer geometry
        geo_transform = self.geo_transform.shifted(buffer_geometry)
        data = self.raster_group.read(buffer_geometry)
        if (data == self.no_data_value).all():
            return False
        data.shape = (1,) + data.shape

        # rasterize the buffer geometry into a raster mask
        mask = np.zeros(data.shape, 'u1')
        dataset_kwargs = {'geo_transform': geo_transform}
        dataset_kwargs.update(self.kwargs)
        with datasources.Layer(buffer_geometry) as layer:
            with datasets.Dataset(mask, **dataset_kwargs) as dataset:
                gdal.RasterizeLayer(dataset, [1], layer, burn_values=[1])

        # rasterize the percentile
        try:
            floor = np.percentile(data[mask.nonzero()], 75)
            if self.floor:
                floor += self.floor
            feature[FLOOR_ATTRIBUTE] = floor
            return True
        except IndexError:
            # no data points at all
            return False

    def rasterize_region(self, index_feature):
        # prepare or abort
        path = self.path(index_feature)
        if exists(path):
            return

        # target array
        target = self.target(index_feature)

        # fetch geometries from postgis
        data_source = self.postgis_source.get_data_source(
            table=self.table,
            geometry=index_feature.geometry(),
        )
        layer = data_source[0]

        # add a column for the floor level
        field_defn = ogr.FieldDefn(FLOOR_ATTRIBUTE, ogr.OFTReal)
        layer.CreateField(field_defn)
        any_computed = False

        # compute floor levels
        feature_count = layer.GetFeatureCount()
        for i in range(feature_count):
            bag_feature = layer[i]
            if self.determine_floor_level(feature=bag_feature):
                layer.SetFeature(bag_feature)
                any_computed = True

        # do not write an empty geotiff
        if not any_computed:
            return

        # rasterize
        options = ['attribute=%s' % FLOOR_ATTRIBUTE]
        gdal.RasterizeLayer(target, [1], layer, options=options)

        # save
        options = ['compress=deflate']
        os.makedirs(dirname(path), exist_ok=True)
        DRIVER_GDAL_GTIFF.CreateCopy(path, target, options=options)


def bag2tif(index_path, part, **kwargs):
    """ Rasterize some postgis tables. """
    index = datasources.PartialDataSource(index_path)
    rasterizer = Rasterizer(**kwargs)

    if part is not None:
        index = index.select(part)

    for feature in index:
        rasterizer.rasterize_region(feature)


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
    parser.add_argument('output_path', metavar='OUTPUT',
                        help='Output folder for result files')
    parser.add_argument('-s', '--host', default='localhost')
    parser.add_argument('-f', '--floor', default=None, type=float)
    parser.add_argument('-u', '--user'),
    parser.add_argument('--part',
                        help='Partial processing source, for example "2/3"')
    return parser


def main():
    """ Call command with args from parser. """
    kwargs = vars(get_parser().parse_args())
    kwargs['password'] = getpass.getpass()
    bag2tif(**kwargs)
