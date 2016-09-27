# (c) Nelen & Schuurmans, see LICENSE.rst.
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import logging
import math
import os

import numpy as np

from raster_tools import gdal
from raster_tools import ogr

logger = logging.getLogger(__name__)


def get_inverse(a, b, c, d):
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
    def __init__(self, geo_transform_tuple):
        """First argument must be a 6-tuple defining a geotransform."""
        super(GeoTransform, self).__init__(geo_transform_tuple)

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

    def scaled(self, f):
        """
        Return shifted geo transform.

        :param f: scale the cellsize by this factor
        """
        p, a, b, q, c, d = self
        return self.__class__([p, a * f, b * f, q, c * f, d * f])

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


class PartialDataSource(object):
    """ Wrap a shapefile. """
    def __init__(self, path):
        self.dataset = ogr.Open(path)
        self.layer = self.dataset[0]

    def __iter__(self):
        total = len(self)
        gdal.TermProgress_nocb(0)
        for count, feature in enumerate(self.layer, 1):
            yield feature
            gdal.TermProgress_nocb(count / total)

    def __len__(self):
        return self.layer.GetFeatureCount()

    def select(self, text):
        """ Return generator of features for text, e.g. '2/5' """
        selected, parts = map(int, text.split('/'))
        size = len(self) / parts
        start = int((selected - 1) * size)
        stop = len(self) if selected == parts else int(selected * size)
        total = stop - start
        gdal.TermProgress_nocb(0)
        for count, fid in enumerate(range(start, stop), 1):
            yield self.layer[fid]
            gdal.TermProgress_nocb(count / total)


class TargetDataSource(object):
    """ Wrap a shapefile, copied from raster-analysis. """
    def __init__(self, path, template_path, attributes):
        # read template
        template_data_source = ogr.Open(template_path)
        template_layer = template_data_source[0]
        template_sr = template_layer.GetSpatialRef()

        # create or replace shape
        driver = ogr.GetDriverByName(b'ESRI Shapefile')
        self.dataset = driver.CreateDataSource(str(path))
        layer_name = os.path.basename(path)
        self.layer = self.dataset.CreateLayer(layer_name, template_sr)

        # Copy field definitions, remember names
        existing = []
        layer_defn = template_layer.GetLayerDefn()
        for i in range(layer_defn.GetFieldCount()):
            field_defn = layer_defn.GetFieldDefn(i)
            existing.append(field_defn.GetName().lower())
            self.layer.CreateField(field_defn)

        # Add extra fields
        for attribute in attributes:
            if attribute.lower() in existing:
                raise NameError(('field named "{}" already '
                                 'exists in template').format(attribute))

            self.layer.CreateField(ogr.FieldDefn(str(attribute), ogr.OFTReal))
        self.layer_defn = self.layer.GetLayerDefn()

    def append(self, geometry, attributes):
        """ Append geometry and attributes as new feature. """
        feature = ogr.Feature(self.layer_defn)
        feature.SetGeometry(geometry)
        for key, value in attributes.items():
            feature[str(key)] = value
        self.layer.CreateFeature(feature)
