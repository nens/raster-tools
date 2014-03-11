# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import logging

from osgeo import gdal
from osgeo import gdal_array
from osgeo import ogr

logger = logging.getLogger(__name__)


def get_geo_transform(feature):
    """ 
    Return a feature's geo_transform.

    Specialized function for ahn2 index. The rounding is because the index
    geometries have small deviations, but at most about a micrometer.
    """
    x1, x2, y1, y2 = feature.geometry().GetEnvelope()
    return round(x1), 0.5, 0.0, round(y2), 0.0, -0.5


def get_geo_transforms(index_path):
    """
    Return dictionary mapping leaf number to geotransform.
    """

    ogr_index_datasource = ogr.Open(index_path)
    ogr_index_layer = ogr_index_datasource[0]

    return {ogr_index_feature[b'BLADNR'][1:]:
            get_geo_transform(ogr_index_feature)
            for ogr_index_feature in ogr_index_layer}


def array2dataset(array):
    """
    Return gdal dataset.
    """
    # Prepare dataset name pointing to array
    datapointer = array.ctypes.data
    bands, lines, pixels = array.shape
    datatypecode = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype.type)
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
    # Acces the array memory as gdal dataset
    dataset = gdal.Open(dataset_name, gdal.GA_Update)
    return dataset
