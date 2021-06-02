# (c) Nelen & Schuurmans, see LICENSE.rst.
# -*- coding: utf-8 -*-

import logging
import math

from osgeo import ogr
import numpy as np

logger = logging.getLogger(__name__)


def get_inverse(a, b, c, d):
    """ Return inverse for a 2 x 2 matrix with elements (a, b), (c, d). """
    D = 1 / (a * d - b * c)
    return d * D, -b * D, -c * D, a * D


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


def aggregate(values, no_data_value, func='mean'):
    """
    Return aggregated array.

    Arrays with uneven dimension sizes will raise an exception.
    """
    func = getattr(np.ma, func)
    result = func(np.ma.masked_values(
        np.dstack([values[0::2, 0::2],
                   values[0::2, 1::2],
                   values[1::2, 0::2],
                   values[1::2, 1::2]]), no_data_value,
    ), 2).astype(values.dtype).filled(no_data_value)
    return {'values': result, 'no_data_value': no_data_value}


def aggregate_uneven(values, no_data_value, func='mean'):
    """ Pad, fold, return. """
    kwargs = {'no_data_value': no_data_value, 'func': func}

    s1, s2 = values.shape
    p1, p2 = s1 % 2, s2 % 2  # required padding to make even-sized

    # quick out for even-sized dimensions
    if not (p1 or p2):
        return aggregate(values, **kwargs)

    # 4-step: a) even section, b) bottom, c) right and d) corner
    result = np.empty(((s1 + p1) / 2, (s2 + p2) / 2), dtype=values.dtype)
    # the even section
    a = aggregate(values[:s1 - p1, :s2 - p2], **kwargs)
    result[:(s1 - p1) / 2, :(s2 - p2) / 2] = a['values']
    if p1:  # bottom row
        b = aggregate(values[-1:, :s2 - p2].repeat(2, axis=0), **kwargs)
        result[-1:, :(s2 - p2) / 2] = b['values']
    if p2:   # right column
        c = aggregate(values[:s1 - p1, -1:].repeat(2, axis=1), **kwargs)
        result[:(s1 - p1) / 2:, -1:] = c['values']
    if p1 and p2:  # corner pixel
        result[-1, -1] = values[-1, -1]
    return {'values': result, 'no_data_value': no_data_value}


class GeoTransform(tuple):
    def shifted(self, geometry, inflate=False):
        """
        Return shifted geo transform.

        :param geometry: geometry to match
        :param inflate: inflate to nearest top-left grid intersection.
        """
        values = list(self)
        index = self.get_indices(geometry, inflate=inflate)[1::-1]
        values[0], values[3] = self.get_coordinates(index)
        return self.__class__(values)

    def scaled(self, w, h):
        """
        Return shifted geo transform.

        :param f: scale the cellsize by this factor
        """
        p, a, b, q, c, d = self
        return self.__class__([p, a * w, b * h, q, c * w, d * h])

    def get_coordinates(self, indices):
        """ Return x, y coordinates.

        :param indices: i, j tuple of integers or arrays.

        i corresponds to the y direction in a non-skew grid.
        """
        p, a, b, q, c, d = self
        i, j = indices
        return p + a * j + b * i, q + c * j + d * i

    def get_indices(self, geometry, inflate=False):
        """
        Return array indices tuple for geometry.

        :param geometry: geometry to subselect
        :param inflate: inflate envelope to grid, to make sure that
            the entire geometry is contained in resulting indices.
        """
        # spatial coordinates
        x1, x2, y1, y2 = geometry.GetEnvelope()

        # inverse transformation
        p, a, b, q, c, d = self
        e, f, g, h = get_inverse(a, b, c, d)

        # apply to envelope corners
        f_lo, f_hi = (math.floor, math.ceil) if inflate else (round, round)

        X1 = int(f_lo(e * (x1 - p) + f * (y2 - q)))
        Y1 = int(f_lo(g * (x1 - p) + h * (y2 - q)))
        X2 = int(f_hi(e * (x2 - p) + f * (y1 - q)))
        Y2 = int(f_hi(g * (x2 - p) + h * (y1 - q)))

        # prevent zero dimensions in case of inflate
        if inflate:
            if X1 == X2:
                X2 += 1
            if Y1 == Y2:
                Y1 -= 1

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
