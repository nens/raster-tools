# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.
"""
Fill depressions in DEM. The landcover data is supposed to be version 1801c.

TODO
- distinguish between shallow and deep depressions
- fix stepping and offsetting in the complex filling step
"""

import os

from scipy import ndimage
import numpy as np

from raster_tools import datasets
from raster_tools import datasources
from raster_tools import groups

from raster_tools import gdal

GTIF = gdal.GetDriverByName(str('gtiff'))
DTYPE = np.dtype('i8, i8')

COURSES = np.array([(64, 128, 1),
                    (32, 0, 2),
                    (16, 8, 4)], 'u1')

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
    maximum = np.finfo(values.dtype).max  # detect the buildings

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
        mask ^ ndimage.binary_erosion(mask, **kwargs),
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
                # result of get_travelled gives unique indices
                indices = upwards[0][unknown], upwards[1][unknown]
            else:
                # unique indices must be attained from the diff
                indices = diff.nonzero()

            mask[indices] = True

        # done when all masked
        if mask.all():
            return

        # determine labeled depressions and surrounding contours
        diff = np.zeros_like(mask)
        label, total = ndimage.label(~mask, **kwargs)
        for count, slices in enumerate(ndimage.find_objects(label), 1):
            slices = tuple(slice(s.start - 1, s.stop + 1) for s in slices)
            depress = (label[slices] == count)

            # determine contour and mark as starting point for next iteration
            dilated = ndimage.binary_dilation(depress, **kwargs)
            contour = dilated ^ depress

            # find contour minimum
            minimum = values[slices][contour].min()

            if minimum == maximum:
                # depression surrounded by buildings
                mask[slices][depress] = True
                continue

            # mark contour as starting point for next iteration
            diff[slices][contour] = True

            # raise inner depression to contour minimum
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
    for step, offset in (100, 0), (100, 25), (100, 50):
        for y in range(offset, 1 + height - step, step):
            for x in range(offset, 1 + width - step, step):
                slices = slice(y, y + step), slice(x, x + step)
                _fill_complex_depressions(
                    unique=False,
                    values=values[slices],
                    mask=None if mask is None else mask[slices],
                )
    # stage 2: complete area - this fills way too large depressions
    # _fill_complex_depressions(values=values, mask=mask, unique=True)


class PitFiller(object):
    def __init__(self, output_path, raster_path, cover_path):
        # paths and source data
        self.output_path = output_path

        # rasters
        if os.path.isdir(raster_path):
            raster_datasets = [gdal.Open(os.path.join(raster_path, path))
                               for path in sorted(os.listdir(raster_path))]
        else:
            raster_datasets = [gdal.Open(raster_path)]
        self.raster_group = groups.Group(*raster_datasets)
        self.cover_group = groups.Group(gdal.Open(cover_path))

        # properties
        self.projection = self.raster_group.projection
        self.geo_transform = self.raster_group.geo_transform
        self.no_data_value = self.raster_group.no_data_value

    def fill(self, feature):
        # target path
        name = feature[str('name')]
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

        # geometries
        inner_geometry = feature.geometry()
        outer_geometry = inner_geometry.Buffer(50)

        # geo transforms
        inner_geo_transform = self.geo_transform.shifted(inner_geometry)
        outer_geo_transform = self.geo_transform.shifted(outer_geometry)

        # data
        values = self.raster_group.read(outer_geometry)
        cover = self.cover_group.read(outer_geometry)

        # create mask where cover refers to water
        mask = np.zeros_like(cover, dtype='b1')
        mask.ravel()[:] = np.in1d(cover, (50, 51, 52, 156, 254))

        # set buildings to maximum dem before directions
        building = np.logical_and(cover > 1, cover < 15)
        maximum = np.finfo(values.dtype).max
        original = values[building]
        values[building] = maximum

        # processing
        fill_simple_depressions(values)
        fill_complex_depressions(values=values, mask=mask)

        # put buildings back in place
        values[building] = original

        # cut out
        slices = outer_geo_transform.get_slices(inner_geometry)
        values = values[slices][np.newaxis]

        # save
        options = ['compress=deflate', 'tiled=yes']
        kwargs = {'projection': self.projection,
                  'geo_transform': inner_geo_transform,
                  'no_data_value': self.no_data_value.item()}

        with datasets.Dataset(values, **kwargs) as dataset:
            GTIF.CreateCopy(path, dataset, options=options)


def fillpits(index_path, part, **kwargs):
    """
    """
    # select some or all polygons
    index = datasources.PartialDataSource(index_path)
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
        help='directory of complementary GDAL rasters.'
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
