# (c) Nelen & Schuurmans, see LICENSE.rst.
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import logging

import numpy as np

from raster_tools import gdal_array
from raster_tools import utils

logger = logging.getLogger(__name__)


class Meta(object):
    def __init__(self, dataset):
        band = dataset.GetRasterBand(1)
        data_type = band.DataType
        numpy_type = gdal_array.GDALTypeCodeToNumericTypeCode(data_type)

        # compared
        self.width = dataset.RasterXSize
        self.height = dataset.RasterYSize
        self.data_type = data_type
        self.projection = dataset.GetProjection()
        self.geo_transform = dataset.GetGeoTransform()

        # not compared
        self.dtype = np.dtype(numpy_type)
        self.no_data_value = numpy_type(band.GetNoDataValue())

    def __eq__(self, other):
        return (self.width == other.width and
                self.height == other.height and
                self.data_type == other.data_type and
                self.projection == other.projection and
                self.geo_transform == other.geo_transform)


class Group(object):
    """
    A group of gdal rasters, automatically merges, and has a more pythonic
    interface.
    """
    def __init__(self, *datasets):
        metas = [Meta(dataset) for dataset in datasets]
        meta = metas[0]
        if not all([meta == m for m in metas]):
            raise ValueError('Incopatible rasters.')

        self.dtype = meta.dtype
        self.width = meta.width
        self.height = meta.height
        self.projection = meta.projection
        self.no_data_value = meta.no_data_value
        self.geo_transform = utils.GeoTransform(meta.geo_transform)

        self.no_data_values = [m.no_data_value for m in metas]
        self.datasets = datasets

    def read(self, bounds):
        """
        Return numpy array.

        bounds: x1, y1, x2, y2 pixel window specifcation, or an ogr geometry

        If the bounds fall outside the dataset, the result is padded
        with no data values.
        """
        # find indices
        try:
            x1, y1, x2, y2 = bounds
        except ValueError:
            x1, y1, x2, y2 = self.geo_transform.get_indices(bounds)

        # overlapping bounds
        w, h = self.width, self.height
        p1 = min(w, max(0, x1))
        q1 = min(h, max(0, y1))
        p2 = min(w, max(0, x2))
        q2 = min(h, max(0, y2))

        # result array plus a view for what's actually inside datasets
        array = np.full((y2 - y1, x2 - x1), self.no_data_value, self.dtype)
        view = array[q1 - y1: q2 - y1, p1 - x1: p2 - x1]

        kwargs = {'xoff': p1, 'yoff': q1, 'xsize': p2 - p1, 'ysize': q2 - q1}
        for dataset, no_data_value in zip(self.datasets, self.no_data_values):
            data = dataset.ReadAsArray(**kwargs)
            index = data != no_data_value
            view[index] = data[index]

        return array
