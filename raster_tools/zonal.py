# -*- coding: utf-8 -*-
"""
Calculate zonal statistics of raster for geometries in a shapefile. The
following stats from the numpy library can be used: min, max, mean,
median, std, var, ptp, etc. A number of additional statistics can be
calculated with this script:

- p<n>: the n-percentile of the array, for example p75
- size: the amount of pixels selected by the feature's geometry
- count: the amount of those pixels containing data (as opposed to nodata)

If the statistic is unsuitable as field name in the target shape, a
different field name can be specified like "the_mean:mean" instead of
simply "mean".
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import re

import numpy as np

from raster_tools import gdal

from raster_tools import groups
from raster_tools import datasets
from raster_tools import datasources


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
        self.no_data_value = self.group.no_data_value.item()

        # these kwargs are constant for the whole group
        self.kwargs = {'projection': self.group.projection,
                       'no_data_value': self.no_data_value}

    def read(self, geometry):
        """
        Return 2-tuple (array, index)

        The 'array' array is already a linear array of elements within
        geometry. The The 'data' array is a boolean array indicating
        wether the corresponding element from 'array' contains data,
        i.e., whose value does not correspond to the no_data_value.
        """
        # determine kwargs to use with GDAL datasets
        kwargs = {'geo_transform': self.geo_transform.shifted(geometry)}
        kwargs.update(self.kwargs)

        # read the data for array
        array_2d = self.group.read(geometry)

        # prepare a mask to select elements that are within geometry
        select_2d = np.zeros(array_2d.shape, dtype='u1')
        with datasources.Layer(geometry) as layer:
            with datasets.Dataset(select_2d[np.newaxis], **kwargs) as dataset:
                gdal.RasterizeLayer(dataset, [1], layer, burn_values=[1])

        # select those elements
        array_1d = array_2d[select_2d.astype('b1')]

        # determine data or no data
        if array_1d.dtype.kind == 'f':
            select_1d = ~np.isclose(array_1d, self.no_data_value)
        else:
            select_1d = ~np.equal(array_1d, self.no_data_value)

        return {'array': array_1d[select_1d], 'size': array_1d.size}

    def analyze(self, feature):
        """ Return attributes to write to the result. """
        # retrieve raster data
        geometry = feature.geometry()
        data = self.read(geometry)
        array = data['array']
        size = data['size']

        # apppend statistics
        attributes = feature.items()
        for column, (action, args) in self.actions.items():
            if action == 'count':
                attributes[column] = array.size
            elif action == 'size':
                attributes[column] = size
            else:
                try:
                    value = getattr(np, action)(array, *args)
                    attributes[column] = round(value.item(), 16)
                except (ValueError, IndexError) as error:
                    template = 'Error getting statistic {} on feature {}: {}'
                    print(template.format(action, feature.GetFID(), error))
                    attributes[column] = np.nan

        return {'geometry': geometry, 'attributes': attributes}


def command(source_path, target_path, raster_paths, statistics, part):
    """ Main """
    # open source datasource
    source = datasources.PartialDataSource(source_path)
    if part is not None:
        source = source.select(part)

    analyzer = Analyzer(statistics=statistics,
                        raster_paths=raster_paths)

    # create target datasource
    target = datasources.TargetDataSource(path=target_path,
                                          template_path=source_path,
                                          attributes=analyzer.actions)

    for feature in source:
        geometry = feature.geometry()
        # print(feature.GetFID())
        if geometry.Area() > 1000000:
            continue
        target.append(**analyzer.analyze(feature))
    return 0


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
