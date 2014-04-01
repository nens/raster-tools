# -*- coding: utf-8 -*-
"""
Create a watershed shapefile from a digital elevation model and a
shapefile with sink points. If a sigma is supplied, additional local
minima in the dem are added to the sink points. If sigma is greater
than zero, the elevation is first smoothed using a gaussian filter with
standard deviation sigma, before finding the local minima.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import math
import sys

from osgeo import ogr
from osgeo import gdal

from scipy import ndimage
import numpy as np

ogr.UseExceptions()
gdal.UseExceptions()
logger = logging.getLogger(__name__)
POLYGON_LAYER = b'polygon'
POLYGON_FIELD = b'polygon'
DRIVER_OGR_SHAPEFILE = ogr.GetDriverByName(b'ESRI Shapefile')
DRIVER_OGR_MEMORY = ogr.GetDriverByName(b'Memory')
DRIVER_GDAL_MEMORY = gdal.GetDriverByName(b'mem')


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        'elevation_path',
        metavar='ELEVATION',
    )
    parser.add_argument(
        'points_path',
        metavar='POINTS',
    )
    parser.add_argument(
        'target_path',
        metavar='TARGET',
    )
    parser.add_argument(
        '-s', '--sigma',
        type=float,
        help='For gaussian smoothing.',
    )
    return parser


def rescale(array):
    """ Rescale array to within min, max. """
    return np.interp(
        array, [array.min(), array.max()], [0, 65534],
    ).astype('u2')


def background(array, sigma):
    """ Return array with marked local minima. """
    if sigma > 0:
        a = ndimage.gaussian_filter(array, sigma)
    else:
        a = array
    m = np.equal(a, ndimage.minimum_filter(a, size=(3, 3)))
    return -1 * m.astype('i2')


def modify(markers, elevation, points):
    """ Modify markers in place. """
    converter = Converter(elevation)
    layer = points[0]
    for feature in layer:
        indices = converter.convert(feature.geometry().GetPoint_2D())
        if indices:
            markers[indices] = feature.GetFID() + 1
    layer.ResetReading()


def polygonize(watershed, elevation):
    # ogr
    datasource = DRIVER_OGR_MEMORY.CreateDataSource('')
    layer = datasource.CreateLayer(POLYGON_LAYER)
    field_defn = ogr.FieldDefn(POLYGON_FIELD, ogr.OFTInteger)
    layer.CreateField(field_defn)

    # gdal
    dataset = DRIVER_GDAL_MEMORY.Create(
        '', elevation.RasterXSize, elevation.RasterYSize, 1, gdal.GDT_Int16,
    )
    dataset.SetProjection(elevation.GetProjection())
    dataset.SetGeoTransform(elevation.GetGeoTransform())
    band = dataset.GetRasterBand(1)
    band.WriteArray(np.where(watershed == -1, 0, watershed))
    band.SetNoDataValue(0)
    mask_band = band.GetMaskBand()

    # polygonize
    gdal.Polygonize(band, mask_band, layer, 0, [], gdal.TermProgress_nocb)
    return datasource


def store(target, points, polygons):
    """
    Create a layer in target with attributes from points and geometries
    from polygons.
    """
    # create target layer with definition from points
    points_layer = points[0]
    layer_defn = points_layer.GetLayerDefn()
    layer = target.CreateLayer(b'watershed', points_layer.GetSpatialRef())
    for i in range(layer_defn.GetFieldCount()):
        layer.CreateField(layer_defn.GetFieldDefn(i))

    # loop polygons in fid order
    polygon_sql = b"SELECT * FROM {layer} ORDER BY {field}".format(
        layer=POLYGON_LAYER, field=POLYGON_FIELD,
    )
    polygon_layer = polygons.ExecuteSQL(polygon_sql)

    # main loop
    for polygon_feature in polygon_layer:
        points_fid = polygon_feature[POLYGON_FIELD] - 1
        points_feature = points_layer[points_fid]
        feature = ogr.Feature(layer_defn)
        for k in points_feature.items():
            feature[k] = points_feature[k]
        feature.SetGeometry(polygon_feature.geometry())
        layer.CreateFeature(feature)


def command(elevation_path, points_path, target_path, sigma):
    """ Call 'm all."""
    target = DRIVER_OGR_SHAPEFILE.CreateDataSource(target_path)

    print('Read elevation')
    elevation = gdal.Open(elevation_path)
    original = elevation.ReadAsArray()
    rescaled = rescale(original)

    print('Mark background')
    if sigma is None:
        markers = np.zeros(original.shape, dtype='i2')
    else:
        markers = background(original, sigma)

    print('Mark points')
    points = ogr.Open(points_path)
    modify(markers=markers, elevation=elevation, points=points)

    print('Calculate watershed')
    watershed = ndimage.watershed_ift(rescaled, markers)
    watershed[watershed == -1] = 0

    print('Polygonize watershed')
    polygons = polygonize(watershed=watershed, elevation=elevation)

    print('Save watershed')
    store(target=target, points=points, polygons=polygons)


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return command(**vars(get_parser().parse_args()))


class Converter(object):
    """ Converts dataset coordinates to gridcoordinates. """
    def __init__(self, dataset):
        self.size = dataset.RasterXSize, dataset.RasterYSize

        p, a, b, q, c, d = dataset.GetGeoTransform()
        self.matrix = np.linalg.inv([(a, b), (c, d)])
        self.origin = p, q

    def index(self, point, origin, delta):
        return int(math.floor(
            delta[0] * (point[0] - origin[0]) +
            delta[1] * (point[1] - origin[1])
        ))

    def convert(self, point):
        """ Points must be of shape 2, N """
        j = self.index(point=point,
                       origin=self.origin,
                       delta=self.matrix[0])
        if not (0 <= j < self.size[0]):
            return
        i = self.index(point=point,
                       origin=self.origin,
                       delta=self.matrix[1])
        if not (0 <= i < self.size[1]):
            return
        return i, j
