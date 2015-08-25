# -*- coding: utf-8 -*-
"""
Interpolate nodata regions in a raster using IDW.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)

import numpy as np
from osgeo import gdal
from osgeo import gdal_array
from osgeo import ogr
from osgeo import osr
from scipy import ndimage

from raster_tools import datasets
from raster_tools import utils

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

GTIF = gdal.GetDriverByName(b'gtiff')

VOID = 0
DATA = 1
ELSE = 2


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
        'raster_path',
        metavar='RASTER',
        help='source GDAL raster dataset with voids'
    )
    parser.add_argument(
        'output_path',
        metavar='OUTPUT',
        help='target folder',
    )
    parser.add_argument(
        '-m', '--mask',
        metavar='MASK',
        dest='mask_path',
        help='shapefile with regions to ignore',
    )
    parser.add_argument(
        '-p', '--part',
        help='partial processing source, for example "2/3"',
    )
    return parser


def perform_idw(source, target, source_mask, target_mask):
    """ Use idw for interpolation. """
    source_index = source_mask.nonzero()
    if len(source_index[0]) == 0:
        return  # no source points for this no data region
    source_points = np.vstack(source_index).transpose()
    source_values = source[source_index]

    target_index = target_mask.nonzero()
    target_points = np.vstack(target_index).transpose()

    sum_of_weights = np.zeros(len(target_points))
    sum_of_weighted_measurements = np.zeros(len(target_points))
    for i in range(len(source_points)):
        distance = np.sqrt(
            np.square(source_points[i] - target_points).sum(1)
        )
        weight = 1 / distance ** 4
        weighted_measurement = source_values[i] * weight
        sum_of_weights += weight
        sum_of_weighted_measurements += weighted_measurement
    target_values = sum_of_weighted_measurements / sum_of_weights
    target[target_index] = target_values


class Grower(object):
    def __init__(self, shape):
        self.shape = shape

    def grow(self, slices):
        """ Grow slices by one, but do not exceed shape dims. """
        return tuple(slice(
            max(0, s.start - 1),
            min(l, s.stop + 1))
            for s, l in zip(slices, self.shape))


class Interpolator(object):
    def __init__(self, mask_path, output_path, raster_dataset, margin=100):
        self.margin = margin
        self.output_path = output_path
        self.raster_dataset = raster_dataset

        self.projection = raster_dataset.GetProjection()
        self.geometry = utils.get_geometry(raster_dataset)
        self.geo_transform = utils.GeoTransform(
            raster_dataset.GetGeoTransform(),
        )

        if mask_path is None:
            self.mask_layer = None
        else:
            self.mask_data_source = ogr.Open(mask_path)
            self.mask_layer = self.mask_data_source[0]

        # no data value
        band = raster_dataset.GetRasterBand(1)
        data_type = band.DataType
        no_data_value = band.GetNoDataValue()
        self.no_data_value = gdal_array.flip_code(data_type)(no_data_value)

    def get_arrays(self, geometry):
        """ Meta is the three class array. """
        # read the data
        window = self.geo_transform.get_window(geometry)
        source = self.raster_dataset.ReadAsArray(**window)
        target = np.ones(source.shape, dtype=source.dtype) * self.no_data_value
        meta = np.where(np.equal(source,
                                 self.no_data_value), VOID, DATA).astype('u1')

        # rasterize the water if mask is available
        if self.mask_layer is not None:
            kwargs = {'geo_transform': self.geo_transform.shifted(geometry)}
            with datasets.Dataset(meta[np.newaxis], **kwargs) as dataset:
                gdal.RasterizeLayer(dataset,
                                    [1],
                                    self.mask_layer,
                                    burn_values=[ELSE])

        return source, target, meta

    def interpolate(self, index_feature):
        # target path
        leaf_number = index_feature[b'BLADNR']
        path = os.path.join(self.output_path,
                            leaf_number[:3],
                            '{}.tif'.format(leaf_number))
        dirname = os.path.dirname(path)
        if os.path.exists(path):
            logger.debug('Target already exists.')
            return
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # geometries
        inner_geometry = index_feature.geometry()
        outer_geometry = (inner_geometry
                          .Buffer(self.margin, 1)
                          .Intersection(self.geometry))
        inner_geo_transform = self.geo_transform.shifted(inner_geometry)
        outer_geo_transform = self.geo_transform.shifted(outer_geometry)

        # arrays
        source, target, meta = self.get_arrays(outer_geometry)
        void_mask = np.equal(meta, VOID)
        data_mask = np.equal(meta, DATA)

        # action
        grower = Grower(shape=meta.shape)
        label, total = ndimage.label(void_mask)
        objects = (grower.grow(o) for o in ndimage.find_objects(label))
        if not total:
            logger.debug('No objects found.')
            return

        for count, slices in enumerate(objects, 1):
            # the masking
            target_mask = np.equal(label[slices], count)
            edge = ndimage.binary_dilation(target_mask) - target_mask
            source_mask = np.logical_and(data_mask[slices], edge)
            # the filling
            perform_idw(source=source[slices],
                        target=target[slices],
                        source_mask=source_mask,
                        target_mask=target_mask)

        # save
        slices = outer_geo_transform.get_slices(inner_geometry)
        if np.equal(target[slices], self.no_data_value).all():
            logger.debug('Nothing filled.')
            return

        kwargs = {'projection': self.projection,
                  'geo_transform': inner_geo_transform,
                  'no_data_value': self.no_data_value.item()}
        with datasets.Dataset(target[slices][np.newaxis], **kwargs) as dataset:
            GTIF.CreateCopy(path, dataset, options=['COMPRESS=DEFLATE'])


def command(index_path, mask_path, raster_path, output_path, part):
    """
    - use label to find the edge of islands of nodata
    - interpolate or take the min from the part of the edges that have data
    - write to output according to index
    """
    # select some or all polygons
    index = utils.PartialDataSource(index_path)
    if part is not None:
        index = index.select(part)

    raster_dataset = gdal.Open(raster_path)

    interpolator = Interpolator(mask_path=mask_path,
                                output_path=output_path,
                                raster_dataset=raster_dataset)

    for feature in index:
        interpolator.interpolate(feature)
    return 0


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr,
                        level=logging.INFO,
                        format='%(message)s')
    try:
        return command(**vars(get_parser().parse_args()))
    except SystemExit:
        raise  # argparse does this
    except:
        logger.exception('An exception has occurred.')
