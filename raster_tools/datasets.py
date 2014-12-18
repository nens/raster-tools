# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from osgeo import gdal
from osgeo import gdal_array

gdal.UseExceptions()


def create(array, geo_transform=None, projection=None, no_data_value=None):
    """
    Create and return a gdal dataset.

    :param array: A numpy array.
    :param geo_transform: 6-tuple of floats
    :param projection: wkt projection string
    :param no_data_value: integer or float

    This is the fastest way to get a gdal dataset from a numpy array, but
    keep a reference to the array around, or a segfault will occur. Also,
    don't forget to call FlushCache() on the dataset after any operation
    that affects the array.
    """
    # prepare dataset name pointing to array
    datapointer = array.ctypes.data
    bands, lines, pixels = array.shape
    datatypecode = gdal_array.NumericTypeCodeToGDALTypeCode(
        array.dtype.type)
    datatype = gdal.GetDataTypeName(datatypecode)
    bandoffset, lineoffset, pixeloffset = array.strides

    dataset_name_template = (
        'MEM:::'
        'DATAPOINTER={datapointer},'
        'PIXELS={pixels},'
        'LINES={lines},'
        'BANDS={bands},'
        'DATATYPE={datatype},'
        'PIXELOFFSET={pixeloffset},'
        'LINEOFFSET={lineoffset},'
        'BANDOFFSET={bandoffset}'
    )
    dataset_name = dataset_name_template.format(
        datapointer=datapointer,
        pixels=pixels,
        lines=lines,
        bands=bands,
        datatype=datatype,
        pixeloffset=pixeloffset,
        lineoffset=lineoffset,
        bandoffset=bandoffset,
    )

    # access the array memory as gdal dataset
    dataset = gdal.Open(dataset_name, gdal.GA_Update)

    # set additional properties from kwargs
    if geo_transform is not None:
        dataset.SetGeoTransform(geo_transform)
    if projection is not None:
        dataset.SetProjection(projection)
    if no_data_value is not None:
        for i in range(len(array)):
            dataset.GetRasterBand(i + 1).SetNoDataValue(no_data_value)

    return dataset


class Dataset(object):
    """
    Usage:
        >>> with Dataset(array) as dataset:
        ...     # do gdal things.
    """
    def __init__(self, array, **kwargs):
        self.array = array
        self.dataset = create(array, **kwargs)

    def __enter__(self):
        return self.dataset

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.dataset.FlushCache()
