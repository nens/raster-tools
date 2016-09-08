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
import sys

import numpy as np

from raster_tools import gdal
from raster_tools import ogr

from raster_tools import groups
from raster_tools import utils

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


def mask(dataset, geometry):
    """ Clip dataset in place to geometry is into dataset. """
    # determine inverse geometry
    x1, x2, y1, y2 = geometry.GetEnvelope()
    wkt = POLYGON.format(x1=x1, y1=y1, x2=x2, y2=y2)
    sr = geometry.GetSpatialReference()
    box = ogr.CreateGeometryFromWkt(wkt, sr)
    mask = box.Difference(geometry)

    # put geometry into temporary layer
    datasource = DRIVER_OGR_MEMORY.CreateDataSource('')
    layer = datasource.CreateLayer(str(''), sr)
    layer_defn = layer.GetLayerDefn()
    feature = ogr.Feature(layer_defn)
    feature.SetGeometry(mask)
    layer.CreateFeature(feature)

    # burn no data
    burn_values = [dataset.GetRasterBand(1).GetNoDataValue()]
    gdal.RasterizeLayer(dataset, [1], layer, burn_values=burn_values)


class Analyzer(object):
    """
    A container that does the computation per feature.
    """
    def __init__(self, output_path, raster_paths, statistics):
        # raster group
        self.group = groups.Group(*map(gdal.Open, raster_paths))

        # prepare statistics gathering
        self.actions = {}  # column_name: func_name, args
        percentile = None
        pattern = re.compile('(p)([0-9]+)')
        for statistic in statistics:
            # allow for different column name
            try:
                column, statistic = statistic.split(':')
            except ValueError:
                column = statistic

            # determine the action
            match = pattern.match(statistic)
            if pattern.match(statistic):
                percentile = int(match.groups()[1])
                actions[column] = 'percentile', [percentile]
            elif statistic == 'value':
                self.actions[column] = 'item', []
            else:
                self.actions[column] = statistic, []

        # keep convenient group properties available
        self.geo_transform = self.group.geo_transform

        self.kwargs = {'projection': self.group.projection,
                       'no_data_value': self.group.no_data_value}

    def analyze(self, feature)
        """ Return attributes to write to the result. """
        geometry = feature.geometry()
        array = self.group.read(geometry)
        geo_transform = self.group.geo_transform.shift(geometry)

        kwargs = {'geo_transform' = self.geo_transform.shift(geometry)}
        kwargs.update(self.kwargs)

        mask = np.zeros(array.shape, dtype='u1')

        with data_sources.Layer(geometry) as layer:
            with dataset.Dataset(mask, **kwargs) as dataset:
                gdal.RasterizeLayer(dataset, [1], layer, burn_values=[1])

        mask_appropriate

   
                

            utils.burn(dataset=dataset, geometry=geometry)
            'no_data_value': self.group.no_data_value,
            'projection': = self.group.projection



        

        
        # get array from group




def command(shape_path, raster_paths, output_path, statistics, part):
    """ Main """
    # shape with source features
    shape = utils.PartialDataSource(shape_path)
    if part is not None:
        index = index.select(part)


    target = utils.TargetDataSource(
        path=output_path,
        template_path=shape_path,
        attributes=actions,
    )

    for source_feature in source_features:
        # retrieve raster data
        geometry = source_feature.geometry()
        kwargs = get_kwargs(geometry)
        data = store.get_data(geometry, **kwargs)
        masked = np.ma.masked_equal(data['values'],
                                    data['no_data_value'])
        compressed = masked.compressed()

        # apppend statistics
        attributes = source_feature.items()
        for column, (action, args) in actions.items():
            try:
                if hasattr(np.ma, action):
                    value = getattr(np.ma, action)(masked, *args)
                else:
                    value = getattr(np, action)(compressed, *args)
                value = np.nan if np.ma.is_masked(value) else value
            except (ValueError, IndexError):
                value = np.nan
            attributes[column] = value

        target.append(geometry=geometry, attributes=attributes)
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
