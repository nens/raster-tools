# (c) Nelen & Schuurmans, see LICENSE.rst.
# -*- coding: utf-8 -*-
"""
This script is the basis for a seamless void-filling procedure for large
coverages. The main principle is to aggregate the raster per quad of
pixels and repeat until only one pixel is left, and then zooming back
in and smoothing at each zoom step.

Optionally a limiting ceiling raster can be supplied to cut the recursion
at an earlier zomlevel. This should be a preaggregated, void-less
dataset. This ceiling dataset can be created from the original dataset
using the aggregate script based on an even tiling (generated with the
reindex script). The results can then be merged using gdal_merge.py and
filled using this script.
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


def fill(values, no_data_value, ceiling):
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
    if ceiling is not None and ceiling.shape == aggregated_shape:
        aggregated = {'values': ceiling, 'no_data_value': no_data_value}
    else:
        aggregated = utils.aggregate_uneven(values=values,
                                            no_data_value=no_data_value)

    filled = fill(ceiling=ceiling, **aggregated)
    zoomed = zoom(filled)[:values.shape[0], :values.shape[1]]
    return np.where(mask, smooth(zoomed), values)


class Filler(object):
    def __init__(self, source_path, ceiling_path):
        # source group
        self.source = groups.Group(gdal.Open(source_path))

        # ceiling group
        if ceiling_path:
            self.ceiling = groups.Group(gdal.Open(ceiling_path))
        else:
            self.ceiling = None

    def fill(self, geometry):
        """ Return dictionary with data and no data value of filling. """
        values = self.source.read(geometry)
        no_data_value = self.source.no_data_value
        if self.ceiling:
            ceiling = self.ceiling.read(geometry)
            if (ceiling == self.ceiling.no_data_value).any():
                # triggers infinite recursion or gives undesired results
                return {'values': values, 'no_data_value': no_data_value}
        else:
            ceiling = None
        result = fill(values=values,
                      ceiling=ceiling,
                      no_data_value=no_data_value)
        result[values != no_data_value] = no_data_value
        return {'values': result, 'no_data_value': no_data_value}


def fillnodata(source_path, target_path, ceiling_path):
    """
    Fill a single raster.
    """
    source_dataset = gdal.Open(source_path)
    geometry = utils.get_geometry(source_dataset)

    filler = Filler(source_path=source_path, ceiling_path=ceiling_path)
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
        '--ceiling', '-c',
        metavar='CEILING',
        dest='ceiling_path',
        help='preaggregated GDAL ceiling raster without voids',
    )
    return parser


def main():
    """ Call fillnodata with args from parser. """
    kwargs = vars(get_parser().parse_args())
    fillnodata(**kwargs)
