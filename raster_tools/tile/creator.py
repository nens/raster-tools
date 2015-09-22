# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.
""" Create a tilemap from a GDAL datasource. """

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import itertools
import logging
import math
import multiprocessing
import os
import sys

from raster_tools import datasets
from raster_tools import gdal
from raster_tools import osr

from osgeo import gdal_array
import numpy as np

logger = logging.getLogger(__name__)

GRA = gdal.GRA_Cubic
LIM = 2 * 6378137 * math.pi
JPG = gdal.GetDriverByName(str('jpeg'))
MEM = gdal.GetDriverByName(str('mem'))
PNG = gdal.GetDriverByName(str('png'))
WKT = osr.GetUserInputAsWKT(str('epsg:3857'))

master = None


def initializer(path):
    """ Assign master dataset to a global variable. """
    global master
    master = gdal.Open(path)


def func(job):
    """ Make tile and return count. """
    tile, count = job
    tile.make()
    return count


def calculate_bbox(dataset):
    # analyze
    w = dataset.RasterXSize
    h = dataset.RasterYSize
    g = dataset.GetGeoTransform()
    coords = map(gdal.ApplyGeoTransform, 4 * [g], 2 * [0, w], [0, 0, h, h])

    # transformation
    source = osr.SpatialReference(dataset.GetProjection())
    target = osr.SpatialReference(WKT)
    ct = osr.CoordinateTransformation(source, target)
    x, y = zip(*ct.TransformPoints(coords))[:2]

    return(min(x), min(y), max(x), max(y))


class Pool(object):
    """ Fake pool. """
    imap = itertools.imap

    def __init__(self, initializer, initargs):
        initializer(*initargs)

    def close(self):
        pass


class Tile(object):
    def __init__(self, base, root, x, y, z):
        self.base = base
        self.root = root

        self.x = x
        self.y = y
        self.z = z

        self.path = os.path.join(root, str(z), str(x), str(y) + '.png')

    def get_geo_transform(self):
        """ Return GeoTransform """
        s = LIM / 2 ** self.z
        p = s * self.x - LIM / 2
        q = LIM / 2 - s * self.y
        a = s / 256
        d = -a
        return p, a, 0, q, 0, d

    def get_subtiles(self):
        """ Return source generator of at most 4 subtiles. """
        z = 1 + self.z
        for dy, dx in itertools.product([0, 1], [0, 1]):
            x = dx + 2 * self.x
            y = dy + 2 * self.y
            yield self.__class__(base=None, root=self.root, x=x, y=y, z=z)

    def as_dataset(self):
        try:
            dataset = gdal.Open(self.path)
        except:
            return

        if dataset.RasterCount == 4:
            # convert transparency to mask
            array = dataset.ReadAsArray().astype('u2')
            array[:3, array[3] == 0] = 256
            dataset = gdal_array.OpenArray(array[:3])

        dataset.SetProjection(WKT)
        dataset.SetGeoTransform(self.get_geo_transform())
        return dataset

    def make(self):
        """ Make tile. """
        # sources
        if self.base:
            sources = [self.as_dataset(), master]
        else:
            sources = [s.as_dataset() for s in self.get_subtiles()]

        # target
        array = np.full((3, 256, 256), 256, dtype='u2')
        kwargs = {'projection': WKT,
                  'no_data_value': 256,
                  'geo_transform': self.get_geo_transform()}

        with datasets.Dataset(array, **kwargs) as target:
            for source in filter(None, sources):
                gdal.ReprojectImage(source, target, None, None, GRA, 0, 0.125)

        # determine type of result
        mask = (array == 256).any(0)[np.newaxis]

        # nothing
        if mask.all():
            return

        # directories
        try:
            os.makedirs(os.path.dirname(self.path))
        except OSError:
            pass  # not necessary

        # png
        if mask.any():
            # convert mask to transparency
            alpha = np.full((1, 256, 256), 255, dtype='u2')
            alpha[mask] = 0
            array = np.vstack([array, alpha])
            target = gdal_array.OpenArray(array.astype('u1'))
            return PNG.CreateCopy(self.path, target)

        # jpg
        target = gdal_array.OpenArray(array.astype('u1'))
        return JPG.CreateCopy(self.path, target, options=['quality=95'])


class Level(object):
    """ A single zoomlevel of tiles. """
    def __init__(self, zoom, root, base, bbox):
        self.root = root
        self.zoom = zoom
        self.bbox = bbox
        self.base = base

    def __len__(self):
        x, y = self.get_xranges()
        return len(x) * len(y)

    def __iter__(self):
        """ Return tile generator. """
        z = self.zoom
        for y, x in itertools.product(*self.get_xranges()):
            yield Tile(base=self.base, root=self.root, x=x, y=y, z=z)

    def get_xranges(self):
        s = LIM / 2 ** self.zoom  # edge length
        h = LIM / 2               # half the earth

        x1 = int(math.floor((h + self.bbox[0]) / s))
        y1 = int(math.floor((h - self.bbox[3]) / s))
        x2 = int(math.ceil((h + self.bbox[2]) / s))
        y2 = int(math.ceil((h - self.bbox[1]) / s))

        return xrange(y1, y2), xrange(x1, x2)


class Pyramid(object):
    def __init__(self, zoom, root, bbox):
        self.bbox = bbox
        self.root = root
        self.zoom = zoom

    def __len__(self):
        return sum(len(l) for l in self)

    def __iter__(self):
        """ Return level generator. """
        bbox = self.bbox

        # yield baselevel
        yield Level(base=True, bbox=bbox, root=self.root, zoom=self.zoom)

        # yield other levels
        for zoom in reversed(range(self.zoom)):
            yield Level(base=False, bbox=bbox, root=self.root, zoom=zoom)


def tiles(source_path, target_path, zoom):
    """ Create tiles. """
    dataset = gdal.Open(source_path)
    bbox = calculate_bbox(dataset)
    pyramid = Pyramid(bbox=bbox, root=target_path, zoom=zoom)

    # separate counts for baselevel and the remaining levels
    total1 = len(iter(pyramid).next())
    total2 = len(pyramid)
    counter = itertools.count(1)

    # disable if multiprocessing is not desired
    Pool = multiprocessing.Pool

    pool = Pool(initializer=initializer, initargs=[source_path])
    for level in pyramid:
        if level.zoom == zoom:
            logger.info('Generating Base Tiles:')
        elif level.zoom == zoom + 1:
            logger.info('Generating Overview Tiles:')
        for count in pool.imap(func, itertools.izip(level, counter)):
            if count > total1:
                progress = (count - total1) / (total2 - total1)
            else:
                progress = count / total1
            gdal.TermProgress_nocb(progress)
    pool.close()

    # remove .aux.xml files created by gdal driver
    logger.info('Remove Useless Files... ')
    for path, dirs, names in os.walk(target_path):
        for name in names:
            if name.endswith('aux.xml'):
                os.remove(os.path.join(path, name))
    logger.info('Done.')


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('source_path', metavar='SOURCE')
    parser.add_argument('target_path', metavar='TARGET')
    parser.add_argument('zoom', metavar='ZOOM', type=int)
    return parser


def main():
    """ Call tiles with args from parser. """
    # logging
    kwargs = vars(get_parser().parse_args())
    if kwargs.pop('verbose'):
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level, format='%(message)s')

    # run or fail
    try:
        tiles(**kwargs)
        return 0
    except:
        logger.exception('An exception has occurred.')
        return 1


if __name__ == '__main__':
    exit(main())
