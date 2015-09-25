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
import io
import itertools
import logging
import math
import multiprocessing
import sys

from raster_tools import datasets
from raster_tools import gdal
from raster_tools import osr

from raster_tools.tile import storages

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


def func(tile):
    """ Make tile and return count. """
    tile.make()
    return tile


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
    """
    Base tile class that allows loading from storage and conversion to
    a gdal dataset.
    """
    def __init__(self, x, y, z, storage):
        self.x = x
        self.y = y
        self.z = z
        self.storage = storage

    def __nonzero__(self):
        return self.data is not None

    def load(self):
        try:
            self.data = self.storage[self.x, self.y, self.z]
        except KeyError:
            self.data = None
        return self

    def get_geo_transform(self):
        """ Return GeoTransform """
        s = LIM / 2 ** self.z
        p = s * self.x - LIM / 2
        q = LIM / 2 - s * self.y
        a = s / 256
        d = -a
        return p, a, 0, q, 0, d

    def as_dataset(self):
        """ Return image as gdal dataset. """
        # convert to rgba array
        array = np.array(Image.open(io.BytesIO(self.data))).transpose(2, 0, 1)
        if len(array) == 3:
            # add alpha
            array = np.vstack([array, np.full_like(array[:1], 255)])

        # return as dataset
        dataset = gdal_array.OpenArray(array)
        dataset.SetProjection(WKT)
        dataset.SetGeoTransform(self.get_geo_transform())
        return dataset


class TargetTile(Tile):
    """
    A tile that can build from sources and save to storage.
    """
    def __init__(self, quality, method,  **kwargs):
        super(TargetTile, self).__init__(**kwargs)
        self.quality = quality
        self.method = method

    def make(self):
        """ Make tile and store data on data attribute. """
        # target
        array = np.zeros((4, 256, 256), dtype='u1')
        kwargs = {'projection': WKT,
                  'geo_transform': self.get_geo_transform()}

        with datasets.Dataset(array, **kwargs) as target:
            gra = GRA[self.method]
            for source in self.get_sources():
                gdal.ReprojectImage(source, target, None, None, gra, 0, 0.125)

        # nothing
        if (array[-1] == 0).all():
            self.data = None
            return

        buf = io.BytesIO()
        if (array[-1] < 255).any():
            image = Image.fromarray(array.transpose(1, 2, 0))
            image.save(buf, format='PNG')
        else:
            image = Image.fromarray(array[:3].transpose(1, 2, 0))
            image.save(buf, format='JPEG', quality=self.quality)
        self.data = buf.getvalue()

    def save(self):
        """ Write data to storage. """
        if self:
            self.storage[self.x, self.y, self.z] = self.data


class BaseTile(TargetTile):
    """
    A tile that has itself and a master as sources.
    """
    def __init__(self, **kwargs):
        """ Same as target tile, but preload. """
        super(BaseTile, self).__init__(**kwargs)

    def get_sources(self):
        yield master
        if self:
            yield self.as_dataset()


class OverviewTile(TargetTile):
    """
    A tile that has its subtiles as sources.
    """
    def __init__(self, **kwargs):
        """ Same as target tile, but store preloaded subtiles. """
        super(OverviewTile, self).__init__(**kwargs)

    def load(self):
        self.subtiles = [s.load() for s in self.get_subtiles()]
        return self

    def get_subtiles(self):
        z = 1 + self.z
        for dy, dx in itertools.product([0, 1], [0, 1]):
            x = dx + 2 * self.x
            y = dy + 2 * self.y
            yield Tile(x=x, y=y, z=z, storage=self.storage)

    def get_sources(self):
        for subtile in self.subtiles:
            if subtile:
                yield subtile.as_dataset()


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
        t = kwargs.pop('tile')
        for y, x in itertools.product(*self.get_xranges()):
            yield t(x=x, y=y, z=z, **kwargs).load()

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
        gra1 = kwargs.pop('gra1')
        gra2 = kwargs.pop('gra2')

        # yield baselevel
        zoom = kwargs.pop('zoom')
        yield Level(tile=BaseTile, zoom=zoom, method=gra1, **kwargs)

        # yield other levels
        for zoom in reversed(range(zoom)):
            yield Level(tile=OverviewTile, zoom=zoom, method=gra2, **kwargs)


def tiles(source_path, target_path, single, **kwargs):
    """ Create tiles. """
    storage = storages.ZipFileStorage(path=target_path)
    # storage = storages.FileStorage(path=target_path)
    dataset = gdal.Open(source_path)
    bbox = BBox(dataset)
    pyramid = Pyramid(storage=storage, bbox=bbox, **kwargs)

    # separate counts for baselevel and the remaining levels
    count = 0
    total1 = len(iter(pyramid).next())
    total2 = len(pyramid)

    # create worker pool
    Pool = DummyPool if single else multiprocessing.Pool
    pool = Pool(initializer=initializer, initargs=[source_path])

    count = 0
    for level_count, level in enumerate(pyramid):

        # progress information
        if level_count == 0:
            logger.info('Generating Base Tiles:')
        elif level_count == 1:
            logger.info('Generating Overview Tiles:')

        for count, tile in enumerate(pool.imap(func, level), count + 1):
            tile.save()

            # progress bar
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
    parser.add_argument('-b', '--base', dest='gra1', default='cubic',
                        help='resampling for base tiles.')
    parser.add_argument('-o', '--overview', dest='gra2', default='cubic',
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
