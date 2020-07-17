# (c) Nelen & Schuurmans, see LICENSE.rst.
# -*- coding: utf-8 -*-

import os

from osgeo import gdal
from osgeo import ogr


class PartialDataSource(object):  # pragma: no cover
    """ Wrap a shapefile. """
    def __init__(self, path):
        self.data_source = ogr.Open(path)
        self.layer = self.data_source[0]

    def __iter__(self):
        total = len(self)
        gdal.TermProgress_nocb(0)
        # this implementation works around an issue in pygdal
        # https://github.com/nextgis/pygdal/issues/31
        for i in range(total):
            yield self.layer[i]
            gdal.TermProgress_nocb((i + 1) / total)

    def __len__(self):
        return self.layer.GetFeatureCount()

    def query(self, geometry):
        """ Return generator of features with geometry as spatial filter. """
        self.layer.SetSpatialFilter(geometry)
        for fid in range(len(self)):
            yield self.layer[fid]
        self.layer.SetSpatialFilter(None)

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


class TargetDataSource(object):  # pragma: no cover
    """ Wrap a shapefile, copied from raster-analysis. """
    def __init__(self, path, template_path, attributes):
        # read template
        template_data_source = ogr.Open(template_path)
        template_layer = template_data_source[0]
        template_sr = template_layer.GetSpatialRef()

        # create or replace shape
        driver = ogr.GetDriverByName('ESRI Shapefile')
        self.dataset = driver.CreateDataSource(path)
        layer_name = os.path.basename(path)
        self.layer = self.dataset.CreateLayer(layer_name, template_sr)

        # copy field definitions, remember names
        existing = []
        layer_defn = template_layer.GetLayerDefn()
        for i in range(layer_defn.GetFieldCount()):
            field_defn = layer_defn.GetFieldDefn(i)
            existing.append(field_defn.GetName().lower())
            self.layer.CreateField(field_defn)

        # add extra fields
        for attribute in attributes:
            if attribute.lower() in existing:
                raise NameError(('field named "{}" already '
                                 'exists in template').format(attribute))
            field_defn = ogr.FieldDefn(str(attribute), ogr.OFTReal)
            field_defn.SetWidth(255)
            field_defn.SetPrecision(15)
            self.layer.CreateField(field_defn)
        self.layer_defn = self.layer.GetLayerDefn()

    def append(self, geometry, attributes):
        """ Append geometry and attributes as new feature. """
        feature = ogr.Feature(self.layer_defn)
        feature.SetGeometry(geometry)
        for key, value in attributes.items():
            feature[str(key)] = value
        self.layer.CreateFeature(feature)


class Layer(object):
    """
    Usage:
        >>> with Layer(geometry) as layer:
        ...     # do ogr things.
    """
    def __init__(self, geometry):
        driver = ogr.GetDriverByName(str('Memory'))
        self.data_source = driver.CreateDataSource('')
        sr = geometry.GetSpatialReference()
        self.layer = self.data_source.CreateLayer(str(''), sr)
        layer_defn = self.layer.GetLayerDefn()
        feature = ogr.Feature(layer_defn)
        feature.SetGeometry(geometry)
        self.layer.CreateFeature(feature)

    def __enter__(self):
        return self.layer

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def iter_layer(layer):
    """ Pygdal has a bug that doesn't allow iterating layers. """
    feature = layer.GetNextFeature()
    while feature is not None:
        yield feature
        feature = layer.GetNextFeature()
    layer.ResetReading()
