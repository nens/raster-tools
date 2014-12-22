# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import logging

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

logger = logging.getLogger(__name__)


def inverse(a, b, c, d):
    """ Return inverse for a 2 x 2 matrix with elements (a, b), (c, d). """
    D = 1 / (a * d - b * c)
    return d * D, -b * D,  -c * D,  a * D


def get_geometry(dataset):
    """
    Return ogr Geometry instance.
    """
    x1, a, b, y2, c, d = dataset.GetGeoTransform()
    x2 = x1 + a * dataset.RasterXSize + b * dataset.RasterYSize
    y1 = y2 + c * dataset.RasterXSize + d * dataset.RasterYSize

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint_2D(x1, y1)
    ring.AddPoint_2D(x2, y1)
    ring.AddPoint_2D(x2, y2)
    ring.AddPoint_2D(x1, y2)
    ring.AddPoint_2D(x1, y1)
    geometry = ogr.Geometry(ogr.wkbPolygon)
    geometry.AddGeometry(ring)
    return geometry


class GeoTransform(tuple):
    def __init__(self, geo_transform_tuple):
        """First argument must be a 6-tuple defining a geotransform."""
        super(GeoTransform, self).__init__(geo_transform_tuple)

    def shifted(self, geometry):
        """
        Return shifted geo transform.

        :param geometry: geometry to match
        """
        values = list(self)
        values[0], x2, y1, values[3] = geometry.GetEnvelope()
        return self.__class__(values)

    def get_indices(self, geometry):
        """
        Return array slices tuple for geometry.

        :param geometry: geometry to subselect
        """
        # spatial coordinates
        x1, x2, y1, y2 = geometry.GetEnvelope()

        # inverse transformation
        p, a, b, q, c, d = self
        e, f, g, h = inverse(a, b, c, d)

        # apply to envelope corners
        X1 = int(round(e * (x1 - p) + f * (y2 - q)))
        Y1 = int(round(g * (x1 - p) + h * (y2 - q)))
        X2 = int(round(e * (x2 - p) + f * (y1 - q)))
        Y2 = int(round(g * (x2 - p) + h * (y1 - q)))

        return X1, Y1, X2, Y2

    def get_slices(self, geometry):
        """
        Return array slices tuple for geometry.

        :param geometry: geometry to subselect
        """
        x1, y1, x2, y2 = self.get_indices(geometry)
        return slice(y1, y2), slice(x1, x2)

    def get_window(self, geometry):
        """
        Return window dictionary for a geometry.

        :param geometry: geometry to subselect
        """
        x1, y1, x2, y2 = self.get_indices(geometry)
        return {'xoff': x1, 'yoff': y1, 'xsize': x2 - x1, 'ysize': y2 - y1}


class PartialDataSource(object):
    """ Wrap a shapefile. """
    def __init__(self, path, part=None):
        self.dataset = ogr.Open(path)
        self.layer = self.dataset[0]
        if part:
            self.count, self.total = map(int, part.split('/'))
        else:
            self.count, self.total = 1, 1

    def __len__(self):
        return self.layer.GetFeatureCount()

    def __iter__(self):
        """ Return generator of features for text, e.g. '2/5' """
        size = len(self) / self.total
        start = int((self.count - 1) * size)
        if self.count == self.total:
            stop = len(self)
        else:
            stop = int(self.count * size)
        for fid in xrange(start, stop):
            yield self.layer[fid]
