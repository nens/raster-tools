# (c) Nelen & Schuurmans, see LICENSE.rst.
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

from math import sqrt, pi

logger = logging.getLogger(__name__)

import numpy as np
from scipy import ndimage
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

from raster_tools import datasets
from raster_tools import utils

from raster_tools import ogr
from raster_tools import gdal
from raster_tools import gdal_array

GTIF = gdal.GetDriverByName(b'gtiff')

VOID = 0
DATA = 1
ELSE = 2

# set the amount of buffering over output geometry in meters to mitigate
# edge effects
RASTER_MARGIN = 100

# use circle approximation to estimate worst possible cross void distance
# below this threshold - above use a more elaborate method because larger
# voids usually have elongated shapes like canals
TARGET_THRESHOLD = 100


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
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='be more verbose',
    )
    return parser


def find_width(mask):
    """
    Determine an approximation for the width of the widest section in
    the mask.
    """
    h, w = mask.shape
    i = np.identity(3, 'b1')

    # create patterns
    backward = np.tile(i, (h // 3 + 1, w // 3 + 1))[:h, :w]
    forward = np.tile(i, (h // 3 + 1, w // 3 + 1))[:h, w - 1::-1]
    vertical = np.tile(np.array([0, 1], 'b1'), (h, w // 2 + 1))[:, :w]
    horizontal = np.tile(np.array([[0], [1]], 'b1'), (h // 2 + 1, w))[:h, :]

    # intersect with mask and find longest line
    maxima = []
    structure = np.ones((3, 3), 'b1')
    for pattern in forward, backward, vertical, horizontal:
        merge = np.logical_and(mask, pattern)
        label, count = ndimage.label(merge, structure=structure)
        maximum = ndimage.sum(merge, label, np.arange(count) + 1).max()
        if pattern is backward or pattern is forward:
            maximum *= sqrt(2)  # the diagonals are longer
        maxima.append(maximum)
    return min(maxima)


def interpolate_void(source_data, target_data, source_mask, target_mask):
    """
    1. Linear interior
    1. NearestNeighbour everywhere, after each smooth
    3. Iterative Uniform Smooth, with decreasing size until some threshold
    """
    # count sources
    source_index = source_mask.nonzero()
    source_count = len(source_index[0])

    if source_count == 0:
        return  # nothing to do

    # count targets
    target_index = target_mask.nonzero()
    target_count = len(target_index[0])

    if target_count == 1:
        # just mean of sources
        target_data[target_mask] = source_data[source_mask].mean()
        return

    # estimate meaningful initial filter size
    target_points = np.vstack(target_index).transpose()

    if target_count < TARGET_THRESHOLD:
        # approximate void by a circle and calculate the diameter
        initial_size = 2 * sqrt(target_count / pi)
    else:
        # determine using stripe pattern method
        initial_size = find_width(target_mask)

    # determine source points and values
    source_points = np.vstack(source_index).transpose()
    source_values = source_data[source_index]

    # determine non-target points
    not_target_mask = ~target_mask
    not_target_index = not_target_mask.nonzero()
    not_target_points = np.vstack(not_target_index).transpose()

    # linear interpolation of inside as initial condition
    linear_interpolator = LinearNDInterpolator(source_points, source_values)
    target_values = linear_interpolator(target_points)

    # nearest neighbour interpolation of the rest
    nearest_interpolator = NearestNDInterpolator(source_points, source_values)
    not_target_values = nearest_interpolator(not_target_points)

    work = np.empty_like(target_data)
    work[target_index] = target_values
    work[not_target_index] = not_target_values

    # the mean of the inputs for points at corners
    missed_mask = np.isnan(work)
    missed_index = missed_mask.nonzero()
    work[missed_index] = source_values.mean()

    for size in sizes(initial_size):
        ndimage.uniform_filter(work, size, output=work)
        work[not_target_index] = not_target_values

    work[not_target_index] = not_target_values
    target_data[target_mask] = work[target_mask]


class Sizes(object):
    def __init__(self):
        f = [3, 3]
        for i in range(62):
            n = f[-2] + f[-1]
            f.append(n if n % 2 else n - 1)
        self.sizes = np.array(f)

    def __call__(self, initial):
        start = np.searchsorted(self.sizes, initial, side='left')
        for i in range(start, -1, -1):
            yield self.sizes[i]


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
    def __init__(self, mask_path, output_path, raster_path):
        self.output_path = output_path
        self.raster_dataset = gdal.Open(raster_path)

        geo_transform = self.raster_dataset.GetGeoTransform()
        self.geo_transform = utils.GeoTransform(geo_transform)
        self.projection = self.raster_dataset.GetProjection()
        self.geometry = utils.get_geometry(self.raster_dataset)

        if mask_path is None:
            self.mask_layer = None
        else:
            self.mask_data_source = ogr.Open(mask_path)
            self.mask_layer = self.mask_data_source[0]

        # no data value
        band = self.raster_dataset.GetRasterBand(1)
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
                          .Buffer(RASTER_MARGIN, 1)
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
            # if count != 61950:
                # continue
            # the masking
            target_mask = np.equal(label[slices], count)
            edge = ndimage.binary_dilation(target_mask) - target_mask
            source_mask = np.logical_and(data_mask[slices], edge)
            # the filling
            interpolate_void(source_mask=source_mask,
                             target_mask=target_mask,
                             source_data=source[slices],
                             target_data=target[slices])
            logger.debug('%2.0f%%', 100 * count / total)

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


def interpolate(index_path, mask_path, raster_path, output_path, part):
    """
    - use label to find the edge of islands of nodata
    - interpolate per void
    - write to output according to index
    """
    # select some or all polygons
    index = utils.PartialDataSource(index_path)
    if part is not None:
        index = index.select(part)

    interpolator = Interpolator(mask_path=mask_path,
                                raster_path=raster_path,
                                output_path=output_path)

    for feature in index:
        interpolator.interpolate(feature)
    return 0


def main():
    """ Call interpolate with args from parser. """
    kwargs = vars(get_parser().parse_args())

    level = logging.DEBUG if kwargs.pop('verbose') else logging.INFO
    logging.basicConfig(**{'level': level,
                           'stream': sys.stderr,
                           'format': '%(message)s'})

    interpolate(**kwargs)


sizes = Sizes()
