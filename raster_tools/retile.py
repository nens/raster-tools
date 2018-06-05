# -*- coding: utf-8 -*-
"""
Retile some large rasters according to an index.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import os

import numpy as np

from raster_tools import gdal
from raster_tools import datasets
from raster_tools import datasources
from raster_tools import groups

driver = gdal.GetDriverByName(str('gtiff'))


class Retiler(object):
    def __init__(self, source_path, target_path):
        """ Init group. """
        if os.path.isdir(source_path):
            raster_datasets = [gdal.Open(os.path.join(source_path, path))
                               for path in sorted(os.listdir(source_path))]
        else:
            raster_datasets = [gdal.Open(source_path)]

        self.group = groups.Group(*raster_datasets)
        self.projection = self.group.projection
        self.geo_transform = self.group.geo_transform
        self.no_data_value = self.group.no_data_value

        self.target_path = target_path

    def retile(self, feature):
        """ Retile to feature. """
        # target path
        name = feature[str('name')]
        path = os.path.join(self.target_path,
                            name[:3],
                            '{}.tif'.format(name))
        if os.path.exists(path):
            return

        # retile
        geometry = feature.geometry()
        geo_transform = self.geo_transform.shifted(geometry)
        try:
            values = self.group.read(geometry)
        except TypeError:
            return
        if (values == self.no_data_value).all():
            return

        # create directory
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass  # no problem

        # save
        kwargs = {'projection': self.projection,
                  'geo_transform': geo_transform,
                  'no_data_value': self.no_data_value.item()}
        options = ['tiled=yes', 'compress=deflate']

        with datasets.Dataset(values[np.newaxis, ...], **kwargs) as dataset:
            driver.CreateCopy(path, dataset, options=options)


def retile(index_path, source_path, target_path, part):
    """ Convert all features. """
    index = datasources.PartialDataSource(index_path)
    if part is not None:
        index = index.select(part)

    retiler = Retiler(source_path=source_path, target_path=target_path)

    for feature in index:
        retiler.retile(feature)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'index_path',
        metavar='INDEX',
        help='shapefile with geometries and names of output tiles',
    )
    parser.add_argument(
        'source_path',
        metavar='SOURCE',
        help='path to source raster or directory with source rasters.'
    )
    parser.add_argument(
        'target_path',
        metavar='TARGET',
        help='target directory',
    )
    parser.add_argument(
        '-p', '--part',
        help='partial processing source, for example "2/3"',
    )
    return parser


def main():
    """ Call hillshade with args from parser. """
    kwargs = vars(get_parser().parse_args())
    retile(**kwargs)
