# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.
"""
Aggregate recursively by taking the mean of quads.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import os

from scipy import ndimage
import numpy as np

from raster_tools import datasets
from raster_tools import utils

from raster_tools import gdal
from raster_tools import gdal_array

GTIF = gdal.GetDriverByName(str('gtiff'))
DTYPE = np.dtype('i8, i8')

COURSES = np.array([(64, 128, 1),
                    (32,   0, 2),
                    (16,   8, 4)], 'u1')

INDICES = COURSES.nonzero()
NUMBERS = COURSES[INDICES][np.newaxis, ...]
OFFSETS = np.array(INDICES).transpose()[np.newaxis] - 1


def fill_simple_depressions(values):
    """ Fill simple depressions in-place. """
    footprint = np.array([(1, 1, 1),
                          (1, 0, 1),
                          (1, 1, 1)], dtype='b1')
    edge = ndimage.minimum_filter(values, footprint=footprint)
    locs = edge > values
    values[locs] = edge[locs]


def calculate_uphill(values):
    """ Return course encoded uphill directions. """
    uphill = np.zeros(values.shape, dtype='u1')
    h, w = values.shape
    for (i, j), k in zip(OFFSETS[0], NUMBERS[0]):
        u1 = max(0, -i)
        u2 = min(h, h - i)
        v1 = max(0, -j)
        v2 = min(w, w - j)
        slices1 = slice(u1, u2), slice(v1, v2)
        slices2 = slice(u1 + i, u2 + i), slice(v1 + j, v2 + j)
        uphill[slices1] += k * (values[slices2] >= values[slices1])
    return uphill


def get_traveled(indices, courses, unique):
    """ Return indices when travelling along courses. """
    # turn indices into points array
    points = np.array(indices).transpose()[:, np.newaxis, :]  # make points

    # determine uphill directions and apply offsets
    encode = courses[indices][:, np.newaxis]                   # which codes
    select = np.bool8(encode & NUMBERS)[..., np.newaxis]       # which courses
    target = (points + select * OFFSETS).reshape(-1, 2)        # apply offsets

    if unique:
        target = np.unique(
            np.ascontiguousarray(target).view(DTYPE)
        ).view(target.dtype).reshape(-1, 2)

    return tuple(target.transpose())                           # return tuple


def _fill_complex_depressions(values, mask=None, unique=False):
    """
    Fill complex depressions in a bottom-up approach, roughly analogous to:

    http://topotools.cr.usgs.gov/pdfs/
    methods_applications_surface_depression_analysis.pdf

    :param values: DEM values
    :param mask: cells defined as not-in-a-depression

    Beware that the values array is modified in-place.
    """
    if mask is None:
        mask = np.zeros(values.shape, dtype='b1')
    else:
        mask = mask.copy()

    # mark edges as not-in-a-depression
    mask[0, :-1] = True
    mask[:-1, -1] = True
    mask[-1, 1:] = True
    mask[1:, 0] = True

    # structure allows for diagonal flow
    kwargs = {'structure': np.ones((3, 3))}

    # initial uphill and indices
    uphill = calculate_uphill(values)
    indices = np.nonzero(
        mask - ndimage.binary_erosion(mask, **kwargs),
    )

    # iterate to raise depressions to pour points
    while True:
        # iterate to find outer contours of depressions
        while True:
            upwards = get_traveled(unique=unique,
                                   courses=uphill,
                                   indices=indices)
            unknown = ~mask[upwards]
            if not unknown.any():
                break

            diff = np.zeros_like(mask)
            diff[upwards[0][unknown], upwards[1][unknown]] = True

            if unique:
                indices = upwards[0][unknown], upwards[1][unknown]
            else:
                indices = diff.nonzero()

            mask[indices] = True

        # done when all masked
        all_masked = mask.all()
        if all_masked:
            return

        # determine labeled depressions and surrounding contours
        diff = np.zeros_like(mask)
        label, total = ndimage.label(~mask, **kwargs)
        for count, slices in enumerate(ndimage.find_objects(label), 1):
            slices = tuple(slice(s.start - 1, s.stop + 1) for s in slices)
            depress = (label[slices] == count)
            dilated = ndimage.binary_dilation(depress, **kwargs)

            # determine contour and mark as starting point for next iteration
            contour = dilated - depress
            diff[slices][contour] = True

            # make contour minimum the new lower limit for the depression
            minimum = values[slices][contour].min()
            values[slices][dilated] = np.maximum(minimum,
                                                 values[slices][dilated])

        # recalculate uphill and take contours as new indices
        uphill = calculate_uphill(values)
        indices = diff.nonzero()


def fill_complex_depressions(values, mask=None):
    """
    Two stage filling.
    """
    # stage 1: blocks of 100 x 100
    height, width = values.shape
    for step, offset in (100, 0), (100, 50):
        for y in range(offset, 1 + height - step, step):
            for x in range(offset, 1 + width - step, step):
                slices = slice(y, y + step), slice(x, x + step)
                _fill_complex_depressions(
                    unique=False,
                    values=values[slices],
                    mask=None if mask is None else mask[slices],
                )
    # stage 2: complete area
    _fill_complex_depressions(values=values, mask=mask, unique=True)


class PitFiller(object):
    def __init__(self, output_path, raster_path, cover_path):
        # paths and source data
        self.output_path = output_path
        self.raster_dataset = gdal.Open(raster_path)
        self.cover_dataset = gdal.Open(cover_path)

        # geospatial reference
        geo_transform = self.raster_dataset.GetGeoTransform()
        self.geo_transform = utils.GeoTransform(geo_transform)
        self.projection = self.raster_dataset.GetProjection()

        # data settings
        band = self.raster_dataset.GetRasterBand(1)
        data_type = band.DataType
        no_data_value = band.GetNoDataValue()
        self.no_data_value = gdal_array.flip_code(data_type)(no_data_value)

    def fill(self, index_feature):
        # target path
        name = index_feature[str('bladnr')]
        path = os.path.join(self.output_path,
                            name[:3],
                            '{}.tif'.format(name))
        if os.path.exists(path):
            return

        # create directory
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass  # no problem

        geometry = index_feature.geometry().Buffer(0)
        geo_transform = self.geo_transform.shifted(geometry)

        # data
        window = self.geo_transform.get_window(geometry)
        values = self.raster_dataset.ReadAsArray(**window)
        cover = self.cover_dataset.ReadAsArray(**window)
        mask = (cover == 144)  # inner water
        no_data_value = self.no_data_value

        # set buildings to maximum dem before directions
        cover = self.cover_dataset.ReadAsArray(**window)
        maximum = np.finfo(values.dtype).max
        building = np.logical_and(cover > 1, cover < 15)
        original = values[building]
        values[building] = maximum

        # processing
        fill_simple_depressions(values)
        fill_complex_depressions(values=values, mask=mask)

        # put buildings back in place
        values[building] = original

        # save
        values = values[np.newaxis]
        options = ['compress=deflate', 'tiled=yes']
        kwargs = {'projection': self.projection,
                  'geo_transform': geo_transform,
                  'no_data_value': no_data_value.item()}

        with datasets.Dataset(values, **kwargs) as dataset:
            GTIF.CreateCopy(path, dataset, options=options)


def fillpits(index_path, part, **kwargs):
    """
    """
    # select some or all polygons
    index = utils.PartialDataSource(index_path)
    if part is not None:
        index = index.select(part)

    pit_filler = PitFiller(**kwargs)

    for feature in index:
        pit_filler.fill(feature)
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
        'raster_path',
        metavar='RASTER',
        help='source GDAL raster that has no voids.'
    )
    parser.add_argument(
        'cover_path',
        metavar='COVER',
        help='functional landuse raster.'
    )
    parser.add_argument(
        'output_path',
        metavar='OUTPUT',
        help='target folder',
    )
    # parser.add_argument(
    #   # '-i', '--iterations',
    #   # type=int, default=6,
    #   # help='partial processing source, for example "2/3"',
    # )
    parser.add_argument(
        '-p', '--part',
        help='partial processing source, for example "2/3"',
    )
    return parser


def main():
    """ Call aggregate with args from parser. """
    kwargs = vars(get_parser().parse_args())
    fillpits(**kwargs)
