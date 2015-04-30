# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
"""
Rasterize zonal statstics (currently only median) into a set of rasters.
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

from raster_tools import datasets
from raster_tools import postgis
from raster_tools import utils

DRIVER_GDAL_GTIFF = gdal.GetDriverByName(b'gtiff')
DRIVER_GDAL_MEM = gdal.GetDriverByName(b'mem')
DRIVER_OGR_MEM = ogr.GetDriverByName(b'memory')

NO_DATA_VALUE = -3.4028234663852886e+38

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

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
        return os.path.join(self.target_dir, leaf[1:4], leaf + '.tif')

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

    def data_source(self, geometry):
        """ Return geometry wrapped as data source. """
        data_source = DRIVER_OGR_MEM.CreateDataSource('')
        layer = data_source.CreateLayer(b'', self.sr)
        layer_defn = layer.GetLayerDefn()
        feature = ogr.Feature(layer_defn)
        feature.SetGeometry(geometry)
        layer.CreateFeature(feature)
        return data_source

    def single(self, feature, target, np_method='median', threshold=None):
        """

        :param feature:
        :param target:
        :param np_method: either median or percentile
        :param threshold: provide value if np_method is percentile
        :return:
        """
        # retrieve raster data
        geo_transform = self.geo_transform.shifted(feature.geometry())
        window = self.geo_transform.get_window(feature.geometry())
        data = self.dataset.ReadAsArray(**window)
        if data is None:
            return
        data.shape = (1,) + data.shape

        # create ogr source with geometry
        data_source = self.data_source(feature.geometry())

        # determine mask
        mask = np.zeros(data.shape, 'u1')
        kwargs = {'geo_transform': geo_transform}
        kwargs.update(self.kwargs)
        with datasets.Dataset(mask, **kwargs) as dataset:
            gdal.RasterizeLayer(dataset, [1], data_source[0], burn_values=[1])

        if np_method == 'median':
            # rasterize the median
            burn = np.median(data[mask.nonzero()])
        elif np_method == 'percentile':
            # rasterize the percentile
            burn = np.percentile(data[mask.nonzero()])
        else:
            raise ValueError("[ERROR] parameter 'np_method' "
                             "must be either 'median' or 'percentile'")
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
            raise Exception('STO')

        # save
        DRIVER_GDAL_GTIFF.CreateCopy(path,
                                     target,
                                     options=['compress=deflate'])


def command(index_path, **kwargs):
    """ Rasterize some postgis tables. """
    rasterizer = Rasterizer(**kwargs)

    data_source = ogr.Open(index_path)
    layer = data_source[0]
    total = layer.GetFeatureCount()
    for count, feature in enumerate(layer, 1):
        rasterizer.rasterize(feature)
        gdal.TermProgress_nocb(count / total)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index_path', metavar='INDEX')
    parser.add_argument('dbname', metavar='DBNAME')
    parser.add_argument('table', metavar='TABLE')
    parser.add_argument('raster_path', metavar='RASTER')
    parser.add_argument('target_dir', metavar='TARGET')
    parser.add_argument('-u', '--user'),
    parser.add_argument('-p', '--password'),
    parser.add_argument('-s', '--host', default='localhost')
    return parser


def main():
    """ Call command with args from parser. """
    import ipdb; ipdb.set_trace()
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
