#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fill nodata and remove foliage from roof elevation data.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import math
import os
import shlex
import string
import subprocess

from scipy import spatial
from scipy import interpolate
import numpy as np

from raster_tools import datasets
from raster_tools import gdal
from raster_tools import ogr
from raster_tools import osr
from raster_tools import vectors

"""

"""
A = +0.25
D = -0.25
NO_DATA_VALUE = np.finfo('f4').min.item()
TIF_DRIVER = gdal.GetDriverByName(str('gtiff'))
MEM_DRIVER = ogr.GetDriverByName(str('Memory'))
OPTIONS = ['compress=deflate', 'tiled=yes']
PROJECTION = osr.GetUserInputAsWKT(str('epsg:28992'))
SR = osr.SpatialReference(PROJECTION)


def clip(kwargs, geometry):
        """ Clip kwargs in place. """
        # do not touch original kwargs
        kwargs = kwargs.copy()
        array = kwargs.pop('array')
        mask = np.ones_like(array, 'u1')

        # create an ogr datasource
        source = MEM_DRIVER.CreateDataSource('')
        layer = source.CreateLayer(str(''), SR)
        defn = layer.GetLayerDefn()
        feature = ogr.Feature(defn)
        feature.SetGeometry(geometry)
        layer.CreateFeature(feature)

        # clip
        with datasets.Dataset(mask, **kwargs) as dataset:
            gdal.RasterizeLayer(dataset, [1], layer, burn_values=[0])

        # alter array with result
        array[mask.astype('b1')] = NO_DATA_VALUE


def rasterize(points, classes):
    """ Create array. """
    px, py, pz = points[classes > 0].transpose()
    x1 = 4 * math.floor(px.min() / 4)
    y1 = 4 * math.floor(py.min() / 4)
    x2 = 4 * math.ceil(px.max() / 4)
    y2 = 4 * math.ceil(py.max() / 4)

    geo_transform = x1, A, 0, y2, 0, D
    array = np.full((4 * (y2 - y1), 4 * (x2 - x1)), NO_DATA_VALUE, 'f4')
    grid = tuple(np.mgrid[y2 + D / 2:y1 + D / 2:D,
                          x1 + A / 2:x2 + A / 2:A][::-1])

    for klass in range(1, classes.max() + 1):
        ix_1d = (classes == klass)
        if ix_1d.sum() < 5:
            continue
        vals = interpolate.griddata(points[ix_1d, :2], points[ix_1d, 2], grid)
        ix_2d = ~np.isnan(vals)
        array[ix_2d] = vals[ix_2d]

    return {'array': array[np.newaxis],
            'projection': PROJECTION,
            'no_data_value': NO_DATA_VALUE,
            'geo_transform': geo_transform}


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
        units = [f[str('unit')] for f in self.layer]
        paths = ' '.join([self.path.format(u) for u in units])
        extent = self._extent(geometry)
        command = self.command.format(paths, extent)
        string = subprocess.check_output(shlex.split(command))
        points = np.fromstring(string, sep=' ').reshape(-1, 3)
        return self._clip(points=points, geometry=geometry)


def classify(points):
    """
    Select any location with enough points in a sphere.
    """
    size = len(points)
    points_2d = points[:, :2]
    tree = spatial.cKDTree(points_2d)
    index = tree.query(points_2d, k=8, distance_upper_bound=1)[1]

    classes = np.zeros(len(points), 'u1')

    valid = (index != size).all(1)
    classes[valid] = 1

    this = points[valid, 2]
    others = points[index[valid], 2]

    # criteria
    crit2 = this < np.percentile(others, 15, 1)
    crit3 = this < others.min(1) + 1
    # crit = np.logical_and(crit1, crit2)
    classes[valid] = np.where(crit2, 2, classes[valid])
    classes[valid] = np.where(~crit3, 3, classes[valid])

    return classes


def parse(points, colors):
    for (x, y, z), (r, g, b) in zip(points, colors):
        yield '{} {} {} {} {} {}'.format(x, y, z, r, g, b)


def roof(index_path, point_path, source_path, target_path):
    fetcher = Fetcher(index_path=index_path, point_path=point_path)
    data_source = ogr.Open(source_path)
    layer = data_source[0]
    for char, feature in zip(string.ascii_letters, layer):
        if char not in 'mn':
            continue
        geometry = feature.geometry()
        geometry = vectors.array2polygon(np.array(geometry.GetPoints()))

        points = fetcher.fetch(geometry)
        classes = np.zeros(len(points), 'u1')

        # classify
        classes = classify(points)
        a, b = 0, 255
        colors = np.array([[b, a, a],
                           [a, b, a],
                           [a, a, b],
                           [b, a, b]], 'u1')[classes]

        # save classified cloud
        text = '\n'.join(parse(points, colors))
        template = 'las2las -stdin -itxt -iparse xyzRGB -o {}.laz'
        command = template.format(char)
        process = subprocess.Popen(shlex.split(command),
                                   stdin=subprocess.PIPE)
        process.communicate(text)

        # kwargs = rasterize(points=points, classes=classes)
        # with datasets.Dataset(**kwargs) as dataset:
            # clip(kwargs=kwargs, geometry=geometry)
            # TIF_DRIVER.CreateCopy(char + '.tif', dataset, options=OPTIONS)
        print(char)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index_path', metavar='INDEX')
    parser.add_argument('point_path', metavar='POINT')
    parser.add_argument('source_path', metavar='SOURCE')
    parser.add_argument('target_path', metavar='TARGET')
    return parser


def main():
    """ Call roof with args from parser. """
    return roof(**vars(get_parser().parse_args()))


if __name__ == '__main__':
    exit(main())
