# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.
"""
Create a tilemap from a GDAL datasource.

Resampling methods can be one of:
bilinear, cubic, average, nearestneighbour, mode, lanczos, cubicspline
"""

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

from PIL import Image
from osgeo import gdal_array
import numpy as np

logger = logging.getLogger(__name__)

LIM = 2 * 6378137 * math.pi
WKT = osr.GetUserInputAsWKT(str('epsg:3857'))
GRA = {n[4:].lower(): getattr(gdal, n)
       for n in dir(gdal) if n.startswith('GRA')}

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


class BBox(object):
    def __init__(self, dataset):
        # analyze
        w = dataset.RasterXSize
        h = dataset.RasterYSize
        g = dataset.GetGeoTransform()
        coords = map(gdal.ApplyGeoTransform, 4 * [g], 2 * [0, w], [0, 0, h, h])

        # transform
        source = osr.SpatialReference(dataset.GetProjection())
        target = osr.SpatialReference(WKT)
        ct = osr.CoordinateTransformation(source, target)
        x, y = zip(*ct.TransformPoints(coords))[:2]

        self.x1, self.y1, self.x2, self.y2 = min(x), min(y), max(x), max(y)


class DummyPool(object):
    """ Dummy pool in case multiprocessing is not used. """
    imap = itertools.imap

    def __init__(self, initializer, initargs):
        initializer(*initargs)


class Tile(object):
    def __init__(self, x, y, z, target_path,
                 quality=None, method=None, base=None):
        self.target_path = target_path
        self.quality = quality
        self.method = method
        self.base = base

        self.x = x
        self.y = y
        self.z = z

        self.path = os.path.join(target_path, str(z), str(x), str(y) + '.png')

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
            yield self.__class__(x=x, y=y, z=z, target_path=self.target_path)

    def as_dataset(self):
        try:
            image = Image.open(self.path)
        except IOError:
            return

        array = np.array(image).transpose(2, 0, 1)

        if len(array) == 3:
            # add alpha
            array = np.vstack([array, np.full_like(array[:1], 255)])

        dataset = gdal_array.OpenArray(array)
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
        array = np.zeros((4, 256, 256), dtype='u1')
        kwargs = {'projection': WKT,
                  'geo_transform': self.get_geo_transform()}

        with datasets.Dataset(array, **kwargs) as target:
            gra = GRA[self.method]
            for source in filter(None, sources):
                gdal.ReprojectImage(source, target, None, None, gra, 0, 0.125)

        # nothing
        if (array[3] == 0).all():
            return

        # directories
        try:
            os.makedirs(os.path.dirname(self.path))
        except OSError:
            pass  # not necessary

        if (array[3] < 255).any():
            # png
            image = Image.fromarray(array.transpose(1, 2, 0))
            return image.save(self.path, format='PNG')

        # jpeg
        image = Image.fromarray(array[:3].transpose(1, 2, 0))
        return image.save(self.path, format='JPEG', quality=self.quality)


class Level(object):
    """ A single zoomlevel of tiles. """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __len__(self):
        x, y = self.get_xranges()
        return len(x) * len(y)

    def __iter__(self):
        """ Return tile generator. """
        kwargs = self.kwargs.copy()
        kwargs.pop('bbox')

        z = kwargs.pop('zoom')
        for y, x in itertools.product(*self.get_xranges()):
            yield Tile(x=x, y=y, z=z, **kwargs)

    def get_xranges(self):
        s = LIM / 2 ** self.kwargs['zoom']  # edge length
        h = LIM / 2                         # half the earth

        bbox = self.kwargs['bbox']

        x1 = int(math.floor((h + bbox.x1) / s))
        y1 = int(math.floor((h - bbox.y2) / s))
        x2 = int(math.ceil((h + bbox.x2) / s))
        y2 = int(math.ceil((h - bbox.y1) / s))

        return xrange(y1, y2), xrange(x1, x2)


class Pyramid(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __len__(self):
        return sum(len(l) for l in self)

    def __iter__(self):
        """ Return level generator. """
        kwargs = self.kwargs.copy()
        method1 = kwargs.pop('method1')
        method2 = kwargs.pop('method2')
        # yield baselevel
        zoom = kwargs.pop('zoom')
        yield Level(base=True, zoom=zoom, method=method1, **kwargs)

        # yield other levels
        for zoom in reversed(range(zoom)):
            yield Level(base=False, zoom=zoom, method=method2, **kwargs)


def tiles(source_path, target_path, single, **kwargs):
    """ Create tiles. """
    dataset = gdal.Open(source_path)
    bbox = BBox(dataset)
    pyramid = Pyramid(target_path=target_path, bbox=bbox, **kwargs)

    # separate counts for baselevel and the remaining levels
    total1 = len(iter(pyramid).next())
    total2 = len(pyramid)
    counter = itertools.count(1)

    # create worker pool
    Pool = DummyPool if single else multiprocessing.Pool
    pool = Pool(initializer=initializer, initargs=[source_path])

    for level_count, level in enumerate(pyramid):
        if level_count == 0:
            logger.info('Generating Base Tiles:')
        elif level_count == 1:
            logger.info('Generating Overview Tiles:')
        for count in pool.imap(func, itertools.izip(level, counter)):
            if count > total1:
                progress = (count - total1) / (total2 - total1)
            else:
                progress = count / total1
            gdal.TermProgress_nocb(progress)


def get_parser():
    """ Return argument parser. """
    FormatterClass1 = argparse.ArgumentDefaultsHelpFormatter
    FormatterClass2 = argparse.RawDescriptionHelpFormatter

    class FormatterClass(FormatterClass1, FormatterClass2):
        pass

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=FormatterClass,
    )
    # positional
    parser.add_argument('-s', '--single', action='store_true',
                        help='disable multiprocessing')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='print debug-level log messages')
    parser.add_argument('-m1', '--method1', default='cubic',
                        help='resampling for base tiles.')
    parser.add_argument('-m2', '--method2', default='cubic',
                        help='resampling for overview tiles.')
    parser.add_argument('-q', '--quality', default=95,
                        type=int, help='JPEG quality for non-edge tiles')
    # optional
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
