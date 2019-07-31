#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract a laz file for a specific polygon from an indexed pointcloud
collection.

Uses external LASTools las2las and las2exe:

- download https://rapidlasso.com/lastools/
- unzip, and in the LAStools directory, run 'make'
- make sure the LAStools/bin is on the path, or symlink to /usr/local/bin
"""

import argparse
import os
import shlex
import subprocess

import numpy as np

from raster_tools import datasets
from raster_tools import datasources
from raster_tools import gdal
from raster_tools import ogr
from raster_tools import osr
from raster_tools import vectors

PROJECTION = osr.GetUserInputAsWKT('epsg:28992')
MEM_DRIVER = ogr.GetDriverByName('Memory')
NO_DATA_VALUE = np.finfo('f4').min.item()
SR = osr.SpatialReference(PROJECTION)


def clip(kwargs, geometry):
    """ Clip kwargs in place. """
    # do not touch original kwargs
    kwargs = kwargs.copy()
    array = kwargs.pop('array')
    mask = np.ones_like(array, 'u1')

    # create an ogr datasource
    source = MEM_DRIVER.CreateDataSource('')
    layer = source.CreateLayer('', SR)
    defn = layer.GetLayerDefn()
    feature = ogr.Feature(defn)
    feature.SetGeometry(geometry)
    layer.CreateFeature(feature)

    # clip
    with datasets.Dataset(mask, **kwargs) as dataset:
        gdal.RasterizeLayer(dataset, [1], layer, burn_values=[0])

    # alter array with result
    array[mask.astype('b1')] = NO_DATA_VALUE


class Fetcher(object):
    def __init__(self, index_path, point_path):
        self.data_source = ogr.Open(index_path)
        self.layer = self.data_source[0]

        # templates
        self.path = os.path.join(point_path, 'u{}.laz')
        self.command = 'las2las -merged -stdout -otxt -i {} -inside {}'

    def _extent(self, geometry):
        x1, x2, y1, y2 = geometry.GetEnvelope()
        return '{} {} {} {}'.format(x1, y1, x2, y2)

    def _clip(self, points, geometry):

        multipoint = vectors.array2multipoint(points)

        # intersection
        intersection = multipoint.Intersection(geometry)

        # to points
        result = np.fromstring(intersection.ExportToWkb()[9:], 'u1')
        array = result.reshape(-1, 29)[:, 5:].copy()
        return array.view('u8').byteswap().view('f8')

    def fetch(self, geometry):
        """ Fetch points using index and las2txt command. """
        self.layer.SetSpatialFilter(geometry)
        units = [f['unit'] for f in datasources.iter_layer(self.layer)]
        paths = ' '.join([self.path.format(u) for u in units])
        extent = self._extent(geometry)
        command = self.command.format(paths, extent)
        string = subprocess.check_output(shlex.split(command))
        points = np.fromstring(string, sep=' ').reshape(-1, 3)
        return self._clip(points=points, geometry=geometry)


def pol2laz(index_path, point_path, source_path, target_path, attribute):
    fetcher = Fetcher(index_path=index_path, point_path=point_path)
    data_source = ogr.Open(source_path)
    layer = data_source[0]

    try:
        os.mkdir(target_path)
    except OSError:
        pass

    for feature in datasources.iter_layer(layer):
        # name
        try:
            name = feature[attribute]
        except ValueError:
            message = 'No attribute "{}" found in selection datasource.'
            print(message.format(attribute))
            exit()

        # fetch points
        geometry = feature.geometry()
        points = fetcher.fetch(geometry)

        # save points
        text = '\n'.join('{} {} {}'.format(x, y, z) for x, y, z in points)
        laz_path = os.path.join(target_path, name + '.laz')
        template = 'las2las -stdin -itxt -iparse xyz -o {}'
        command = template.format(laz_path)
        process = subprocess.Popen(
	    shlex.split(command),
            stdin=subprocess.PIPE,
            universal_newlines=True,
        )
        process.communicate(text)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index_path', metavar='INDEX')
    parser.add_argument('point_path', metavar='POINT')
    parser.add_argument('source_path', metavar='SOURCE')
    parser.add_argument('target_path', metavar='TARGET')
    parser.add_argument('-a', '--attribute', default='name',
                        help='attribute for naming result laz files')
    return parser


def main():
    """ Call pol2laz with args from parser. """
    return pol2laz(**vars(get_parser().parse_args()))


if __name__ == '__main__':
    exit(main())
