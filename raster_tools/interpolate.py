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
import math
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


def find_radius(array):
    """
    s = 7, 9
    i = np.identity(3, 'u1')
    # == and ||
    np.tile(np.array([[0], [1]], 'u1'), (s[0] // 2 + 1, s[1]))[:s[0], :]
    np.tile(np.array([0, 1], 'u1'), (s[0], s[1] // 2 + 1))[:, :s[1]]
    # \\ and //:
    np.tile(i, (s[0] // 3 + 1, s[1] // 3 + 1))[:s[0], :s[1]]
    np.tile(i, (s[0] // 3 + 1, s[1] // 3 + 1))[:s[0], s[1] -1::-1]
    """


def select_sparse(mask, spacer):
    """ Return index. """
    sparse = np.zeros_like(mask)
    sparse[::spacer, ::spacer] = True
    return np.logical_and(mask, sparse).nonzero()


def interpolate_points():
    """
    Return target values as array. The radius parameter selects points
    in a circle to use.

    - take into account radius # done that.
    - take into account direction  # now what, it iterates per weight over all points in the collection! Thats n^2 computation per 
    - take into account slope

    - Think about algorthm for finding all points in a search radius:
        - ckdtree?
        - or 'manual'?i

    - Need to set this up with 2d arrays, relating the points to each
      other for eacht resulting target point. It will be slow for the
      large voids, but the two-phasing makes up for that.
    """
    result = np.empty(len(target))

    for index, point in enumerate(target):
        distance = np.sqrt(np.square(point - points).sum(1))

        # define pieces for weighting function
        piece1 = np.less_equal(distance, radius / 3)
        piece2 = np.logical_and(np.less(radius / 3, distance),
                                np.less_equal(distance, radius))
        pieces = np.logical_or(piece1, piece2)

        # evaluate weighting function
        weight = np.empty_like(distance)
        weight[piece1] = 1 / distance[piece1]
        weight[piece2] = (27 / (4 * radius) *
                          np.square(distance[piece2] / radius - 1))

        # evaluate interpolation function
        weight = np.square(weight[pieces])
        result[index] = (weight * values[pieces]).sum() / weight.sum()

    return result


def interpolate_void(source_data, target_data, source_mask, target_mask):
    """
    Does two phase interpolation if voids are too large for immediate
    interpolation.

    Note that the masks are copies, but the datas are views.
    
    We are going to do the calculation in square groups of points, using the manhattan distance to select the sources.
    """
    source_index = source_mask.nonzero()
    source_points_count = len(source_index[0])
    if source_points_count == 0:
        return

    # radius that would even cross a circular void
    target_index = target_mask.nonzero()
    target_points_count = len(target_index[0])
    worst_radius = math.sqrt(target_points_count / math.pi)

    spacer = 9                                # for extra source points
    radius = spacer * math.sqrt(7 / math.pi)  # when sufficient

    if worst_radius > radius:
        # add sources add regularly spaced pixels
        helper_index = select_sparse(mask=target_mask, spacer=spacer)

        points = np.vstack(source_index).transpose()
        values = source_data[source_index]
        target = np.vstack(helper_index).transpose()

        result = interpolate_points(points=points, values=values,
                                    target=target, radius=worst_radius)

        # the result takes the role of source instead of target
        source_data[helper_index] = result
        source_mask[helper_index] = True
        target_data[helper_index] = result
        target_mask[helper_index] = False

        ma = np.ma.masked_values(target_data, target_data.min())
        imshow(ma, interpolation='none')
        savefig('idw1.png')
        clf()
    else:
        return

    source_index = source_mask.nonzero()
    target_index = target_mask.nonzero()

    points = np.vstack(source_index).transpose()
    values = source_data[source_index]
    target = np.vstack(target_index).transpose()

    target_data[target_index] = interpolate_points(
        points=points, values=values, target=target, radius=radius,
    )

    ma = np.ma.masked_values(target_data, target_data.min())
    imshow(ma, interpolation='none')
    savefig('idw2.png')
    clf()
    exit()


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
        target = np.empty_like(source)
        target.fill(self.no_data_value)
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
        if os.path.exists(path):
            logger.debug('Target already exists.')
            return

        # create directory
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass  # no problem

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
            interpolate_void(source_data=source[slices],
                             target_data=target[slices],
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


from pylab import *
