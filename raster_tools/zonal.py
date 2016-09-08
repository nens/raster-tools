# -*- coding: utf-8 -*-
"""
Calculate zonal statistics of raster store for a shapefile.

Special stats worth mentioning are 'count' (the amount of pixels
with data), 'size' (the total amount of pixels) and 'p<n>' (the
n-percentile). If the statistic is unsuitable as field name in the target
shape, a different field name can be specified like "myfield:count"
instead of simply "count".
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import math
import re

import numpy as np

from raster_tools import gdal

from raster_tools import utils
from raster_tools import groups
from raster_tools import datasets
from raster_tools import data_sources

logger = logging.getLogger(__name__)

POLYGON = 'POLYGON (({x1} {y1},{x2} {y1},{x2} {y2},{x1} {y2},{x1} {y1}))'


def get_masked(values, no_data_value, copy=False):
    """ Return values as masked array. """
    kind = values.dtype.type
    masked = np.ma.masked_values if kind == 'f' else np.ma.masked_equal
    return masked(values, no_data_value, copy=copy)


def get_kwargs(geometry):
    """ Return get_data_kwargs based on ahn2 resolution. """
    name = geometry.GetGeometryName()
    if name == 'POINT':
        return {}
    if name == 'LINESTRING':
        size = int(math.ceil(geometry.Length() / 0.5))
        return {'size': size}
    if name == 'POLYGON':
        x1, x2, y1, y2 = geometry.GetEnvelope()
        width = int(math.ceil((x2 - x1) / 0.5))
        height = int(math.ceil((y2 - y1) / 0.5))
        return {'width': width, 'height': height}


class Analyzer(object):
    """
    A container that does the computation per feature.
    """
    def __init__(self, raster_paths, statistics):
        # raster group
        self.group = groups.Group(*map(gdal.Open, raster_paths))

        # prepare statistics gathering
        self.actions = {}  # column_name: func_name, args
        percentile = None
        pattern = re.compile('(p)([0-9]+)')
        for statistic in statistics:
            # allow for different column name
            if ':' in statistic:
                column, statistic = statistic.split(':')
            else:
                column = statistic

            # determine the action
            match = pattern.match(statistic)
            if pattern.match(statistic):
                percentile = int(match.groups()[1])
                self.actions[column] = 'percentile', [percentile]
            else:
                self.actions[column] = statistic, []

        # keep convenient group properties available
        self.geo_transform = self.group.geo_transform
        self.no_data_value = self.group.no_data_value

        # these kwargs are constant for the whole group
        self.kwargs = {'projection': self.group.projection,
                       'no_data_value': self.group.no_data_value}

    def read(self, geometry):
        """
        Return 2-tuple (array, index)

        The 'array' array is already a linear array of elements within
        geometry. The The 'data' array is a boolean array indicating
        wether the corresponding element from 'array' contains data,
        i.e., whose value does not correspond to the no_data_value.
        """
        # determine kwargs to use with GDAL datasets
        kwargs = {'geo_transform': self.geo_transform.shift(geometry)}
        kwargs.update(self.kwargs)

        # read the data for array
        array_2d = self.group.read(geometry)

        # prepare a mask to select elements that are within geometry
        select = np.zeros(array_2d.shape, dtype='u1')
        with data_sources.Layer(geometry) as layer:
            with datasets.Dataset(select[np.newaxis], **kwargs) as dataset:
                gdal.RasterizeLayer(dataset, [1], layer, burn_values=[1])

        # select those elements
        array = array_2d[select.astype('b1')]

        # determine data or no data
        if array_2d.dtype.kind == 'f':
            data = ~np.isclose(array, self.no_data_value)
        else:
            data = ~np.equal(array, self.no_data_value)

        return array, data

    def analyze(self, feature):
        """ Return attributes to write to the result. """
        # retrieve raster data
        geometry = feature.geometry()
        array, data = self.read(geometry)

        # apppend statistics
        attributes = feature.items()
        for column, (action, args) in self.actions.items():
            attributes[column] = getattr(np, action)(array, *args)

        return {'geometry': geometry, 'attributes': attributes}


def command(source_path, target_path, raster_paths, statistics, part):
    """ Main """
    # open source datasource
    source = utils.PartialDataSource(source_path)
    if part is not None:
        source = source.select(part)

    analyzer = Analyzer(statistics=statistics,
                        raster_paths=raster_paths)

    # create target datasource
    target = utils.TargetDataSource(path=target_path,
                                    template_path=source_path,
                                    attributes=analyzer.actions)

    for feature in source:
        target.append(**analyzer.analyze(feature))
    return 0


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        'source_path',
        metavar='SOURCE',
        help='Path to shapefile with source features.',
    )
    parser.add_argument(
        'target_path',
        metavar='TARGET',
        help='Path to shapefile with target features.',
    )
    parser.add_argument(
        '-r', '--rasters',
        metavar='RASTER',
        dest='raster_paths',
        nargs='+',
        help='Paths to GDAL dataset(s). Multiple datasets will be stacked.',
    )
    parser.add_argument(
        '-s', '--statistics',
        metavar='STAT',
        nargs='+',
        default=('max', 'mean', 'min', 'stddev'),
        help=('One or more stastics to compute per'
              ' feature, for example: "value median p90".'),
    )
    parser.add_argument(
        '-p', '--part',
        help='Partial processing source, for example "2/3"',
    )
    return parser


def main():
    """ Call command with args from parser. """
    return command(**vars(get_parser().parse_args()))
