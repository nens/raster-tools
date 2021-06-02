#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fill nodata and remove foliage from roof elevation data.
"""

import argparse
import math
import os
import shlex
import string
import subprocess

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from scipy import interpolate
from scipy import sparse
from scipy import spatial
from scipy.sparse import csgraph
import numpy as np

from raster_tools import datasets
from raster_tools import vectors

A = +0.25
D = -0.25
NO_DATA_VALUE = np.finfo('f4').min.item()
TIF_DRIVER = gdal.GetDriverByName('gtiff')
MEM_DRIVER = ogr.GetDriverByName('Memory')
OPTIONS = ['compress=deflate', 'tiled=yes']
PROJECTION = osr.GetUserInputAsWKT('epsg:28992')
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


def rasterize(geometry, points):
    """ Create array. """
    envelope = geometry.GetEnvelope()
    # px, py, pz = points.transpose()
    x1 = 4 * math.floor(envelope[0] / 4)
    y1 = 4 * math.floor(envelope[2] / 4)
    x2 = 4 * math.ceil(envelope[1] / 4)
    y2 = 4 * math.ceil(envelope[3] / 4)

    geo_transform = x1, A, 0, y2, 0, D
    array = np.full((4 * (y2 - y1), 4 * (x2 - x1)), NO_DATA_VALUE, 'f4')
    grid = tuple(np.mgrid[y2 + D / 2:y1 + D / 2:D,
                          x1 + A / 2:x2 + A / 2:A][::-1])

    # interpolate
    args = points[:, :2], points[:, 2], grid
    linear = interpolate.griddata(*args, method='linear')
    nearest = interpolate.griddata(*args, method='nearest')
    array = np.where(np.isnan(linear), nearest, linear).astype('f4')

    # clip and return
    kwargs = {
        'array': array[np.newaxis],
        'projection': PROJECTION,
        'no_data_value': NO_DATA_VALUE,
        'geo_transform': geo_transform,
    }
    clip(kwargs=kwargs, geometry=geometry)
    return kwargs


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
        units = [f['unit'] for f in self.layer]
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
    if size < 900:
        # this better be some (local) density
        return np.ones(size, 'u1')

    # use spatial to find near neighbours
    links = 4    # connect to how many points
    limit = 0.5  # connect only within distance
    tree = spatial.cKDTree(points)
    dist, index = tree.query(points, k=links)

    # determine valid links
    start = np.arange(size).repeat(links)
    stop = index.ravel()
    rdist = dist.ravel()
    select = np.logical_and(rdist < limit, start != stop)

    # create sparse matrix and find components
    data = np.ones(links * size, 'b1')
    matrix = sparse.csr_matrix(
        (data[select], (start[select], stop[select])), shape=(size, size),
    )
    comps, label = csgraph.connected_components(matrix, directed=False)

    count = np.bincount(label)
    classes = np.zeros(size, 'u1')
    classes[label == count.argmax()] = 1
    return classes


def parse(points, colors):
    for (x, y, z), (r, g, b) in zip(points, colors):
        yield '{} {} {} {} {} {}'.format(x, y, z, r, g, b)


def roof(index_path, point_path, source_path, target_path):
    fetcher = Fetcher(index_path=index_path, point_path=point_path)
    data_source = ogr.Open(source_path)
    layer = data_source[0]

    try:
        os.mkdir(target_path)
    except OSError:
        pass

    for char, feature in zip(string.ascii_letters, layer):
        # if char not in 'mn':
        #     continue
        geometry = feature.geometry()
        points = fetcher.fetch(geometry)

        # classify
        classes = classify(points)
        a, b = 0, 255
        colors = np.array([[b, a, a],
                           [a, b, a],
                           [a, a, b],
                           [b, a, b]], 'u1')[classes]

        # save classified cloud
        text = '\n'.join(parse(points, colors))
        laz_path = os.path.join(target_path, char + '.laz')
        template = 'las2las -stdin -itxt -iparse xyzRGB -o {}'
        command = template.format(laz_path)
        process = subprocess.Popen(shlex.split(command),
                                   stdin=subprocess.PIPE)
        process.communicate(text)

        # save tif
        tif_path = os.path.join(target_path, char + '.tif')
        points = points[classes.astype('b1')]
        kwargs = rasterize(points=points, geometry=geometry)
        with datasets.Dataset(**kwargs) as dataset:
            TIF_DRIVER.CreateCopy(tif_path, dataset, options=OPTIONS)
        print(char, len(points))


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
