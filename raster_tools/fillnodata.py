# (c) Nelen & Schuurmans, see LICENSE.rst.
# -*- coding: utf-8 -*-
"""
Recursive filling of raster voids for a single raster.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse

import numpy as np
from scipy import ndimage

from raster_tools import datasets
from raster_tools import groups
from raster_tools import utils

from raster_tools import gdal

GTIF = gdal.GetDriverByName(str('gtiff'))
OPTIONS = ['compress=deflate', 'tiled=yes']
KERNEL = np.array([[0.0625, 0.1250,  0.0625],
                   [0.1250, 0.2500,  0.1250],
                   [0.0625, 0.1250,  0.0625]])


def zoom(values):
    """ Return zoomed array. """
    return values.repeat(2, axis=0).repeat(2, axis=1)


def smooth(values):
    """ Two-step uniform for symmetric smoothing. """
    return ndimage.correlate(values, KERNEL)


def fill(values, no_data_value, border):
    """
    Fill must return a filled array. It does so by aggregating, requesting
    a fill for that, and zooming back. After zooming back, it smooths
    the filled values and returns.
    """
    mask = values == no_data_value
    if not mask.any():
        # this should end the recursion
        return values

    # aggregate
    aggregated_shape = values.shape[0] / 2, values.shape[1] / 2
    if border is not None and border.shape == aggregated_shape:
        aggregated = {'values': border, 'no_data_value': no_data_value}
    else:
        aggregated = utils.aggregate_uneven(values=values,
                                            no_data_value=no_data_value)

    filled = fill(border=border, **aggregated)
    zoomed = zoom(filled)[:values.shape[0], :values.shape[1]]
    return np.where(mask, smooth(zoomed), values)


class Filler(object):
    def __init__(self, source_path, border_path):
        # source group
        self.source = groups.Group(gdal.Open(source_path))

        # border group
        if border_path:
            self.border = groups.Group(gdal.Open(border_path))
        else:
            self.border = None

    def fill(self, geometry):
        """ Return dictionary with data and no data value of filling. """
        values = self.source.read(geometry)
        no_data_value = self.source.no_data_value
        if self.border:
            border = self.border.read(geometry)
            if (border == self.border.no_data_value).any():
                # triggers infinite recursion or gives undesired results
                return {'values': values, 'no_data_value': no_data_value}
        else:
            border = None
        result = fill(values=values,
                      border=border,
                      no_data_value=no_data_value)
        result[values != no_data_value] = no_data_value
        return {'values': result, 'no_data_value': no_data_value}


def fillnodata(source_path, target_path, border_path):
    """
    Fill a single raster.
    """
    source_dataset = gdal.Open(source_path)
    geometry = utils.get_geometry(source_dataset)

    filler = Filler(source_path=source_path, border_path=border_path)
    result = filler.fill(geometry)

    kwargs = {'projection': source_dataset.GetProjection(),
              'geo_transform': source_dataset.GetGeoTransform()}
    kwargs['array'] = result['values'][np.newaxis]
    kwargs['no_data_value'] = result['no_data_value'].item()

    with datasets.Dataset(**kwargs) as dataset:
        GTIF.CreateCopy(target_path, dataset, options=OPTIONS)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        'source_path',
        metavar='RASTER',
        help='source GDAL raster dataset with voids'
    )
    parser.add_argument(
        'target_path',
        metavar='OUTPUT',
        help='target GDAL raster without voids',
    )
    parser.add_argument(
        '--border', '-b',
        metavar='BORDER',
        dest='border_path',
        help='preaggregated GDAL border raster without voids',
    )
    return parser


def main():
    """ Call fillnodata with args from parser. """
    kwargs = vars(get_parser().parse_args())
    fillnodata(**kwargs)
