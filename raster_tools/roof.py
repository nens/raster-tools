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
import collections
import math
import os
import shlex
import string
import subprocess

from scipy import spatial
import numpy as np

# from raster_tools import datasets
from raster_tools import gdal
from raster_tools import ogr
from raster_tools import osr
from raster_tools import vectors

"""

"""
WIDTH = 0.5
HEIGHT = 0.5
NO_DATA_VALUE = np.finfo('f4').min.item()
DRIVER = gdal.GetDriverByName(str('gtiff'))
OPTIONS = ['compress=deflate', 'tiled=yes']
PROJECTION = osr.GetUserInputAsWKT(str('epsg:28992'))


def rasterize(points):
    """ Create array. """
    xmin, ymin = points[:, :2].min(0)
    xmax, ymax = points[:, :2].max(0)

    p = math.floor(xmin / WIDTH) * WIDTH
    q = math.floor(ymax / HEIGHT) * HEIGHT

    geo_transform = p, WIDTH, 0, q, 0, -HEIGHT

    indices = np.empty((len(points), 3), 'u4')
    indices[:, 2] = (points[:, 0] - p) / WIDTH
    indices[:, 1] = (q - points[:, 1]) / HEIGHT

    order = indices.view('u4,u4,u4').argsort(order=['f1', 'f2'], axis=0)[:, 0]
    indices = indices[order]

    indices[0, 0] = 0
    py, px = indices[0, 1:]
    for i in range(1, len(indices)):
        same1 = indices[i, 1] == indices[i - 1, 1]
        same2 = indices[i, 2] == indices[i - 1, 2]
        if same1 and same2:
            indices[i, 0] = indices[i - 1, 0] + 1
        else:
            indices[i, 0] = 0

    array = np.full(indices.max(0) + 1, NO_DATA_VALUE)
    array[tuple(indices.transpose())] = points[:, 2][order]
    array = np.ma.masked_values(array, NO_DATA_VALUE)

    return {'array': array,
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
        polygon = vectors.array2polygon(np.array(geometry.GetPoints()))
        intersection = multipoint.Intersection(polygon)

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
    columns = collections.defaultdict(list)
    for i, (x, y, z) in enumerate(points):
        columns[int(x), int(y)].append(i)

    select = np.zeros(len(points), 'b1')
    for l in columns.values():
        z = points[l, 2]  # z values in this patch
        try:
            select[l] = [z < z[z.argsort()[8]] + 1]  # within range of min
        except IndexError:
            select[l] = True
    return select


def planify(points, select):
    p = points[select]
    o = p.min(0)
    p -= o
    t = spatial.Delaunay(p[:, :2])
    s0, s1, s2 = t.simplices.transpose()
    n = np.cross(p[s1] - p[s0], p[s2] - p[s0])
    n /= np.linalg.norm(n, axis=1)[:, np.newaxis]

    d = -(n * p[s0]).sum(1)

    q = np.concatenate([n, (d / d.std())[:, np.newaxis]], axis=1)

    tree = spatial.cKDTree(q)

    # query tree and take mean of some region
    i = tree.query(q, k=20)[0].sum(1).argmin()
    j = tree.query(q[i], k=10)[1]
    try:
        (a, b, c), d = n[j].mean(0), d[j].mean()
    except IndexError:
        return np.zeros(len(select), 'b1')

    x, y, z = p.transpose()
    e = a * x + b * y + c * z + d

    plane = np.zeros(len(select), 'b1')
    plane[select] = np.abs(e) < 0.2
    return plane


def parse(points, colors):
    for (x, y, z), (r, g, b) in zip(points, colors):
        yield '{} {} {} {} {} {}'.format(x, y, z, r, g, b)


def roof(index_path, point_path, source_path, target_path):
    fetcher = Fetcher(index_path=index_path, point_path=point_path)
    data_source = ogr.Open(source_path)
    layer = data_source[0]
    for char, feature in zip(string.ascii_letters, layer):
        # if char not in 'm':
            # continue
        geometry = feature.geometry()
        points = fetcher.fetch(geometry)
        colors = np.tile([0, 0, 255], (len(points), 1))

        # remove foliage
        select = classify(points)
        colors[select] = 0, 255, 0

        tuples = (
            (255, 0, 0),
            (255, 255, 0),
            (255, 255, 255),
            (0, 255, 255),
            (255, 0, 255),
        )

        for t in tuples:
            # mark plane
            plane = planify(points=points, select=select)
            colors[plane] = t
            select[plane] = False

        text = '\n'.join(parse(points, colors))
        template = 'las2las -stdin -itxt -iparse xyzRGB -o {}.laz'
        command = template.format(char)
        process = subprocess.Popen(shlex.split(command),
                                   stdin=subprocess.PIPE)
        process.communicate(text)
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
