#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Subtract all polygons in SHAPE2 from all polygons in SHAPE1, and
place them in a new shape TARGET.
"""
import argparse
import logging
import os
import sys

from osgeo import ogr

logger = logging.getLogger(__name__)


def command(shape1_path, shape2_path, target_path):
    shape1 = ogr.Open(shape1_path)
    layer1 = shape1[0]
    shape2 = ogr.Open(shape2_path)
    layer2 = shape2[0]

    # delete any existing target
    driver = ogr.GetDriverByName(b'ESRI Shapefile')
    try:
        driver.DeleteDataSource(str(target_path))
    except RuntimeError:
        pass

    # prepare target dataset
    target_sr = layer1.GetSpatialRef()
    target_shape = driver.CreateDataSource(str(target_path))
    target_layer_name = os.path.basename(target_path)
    target_layer = target_shape.CreateLayer(target_layer_name, target_sr)
    target_layer_defn = layer1.GetLayerDefn()
    for i in range(target_layer_defn.GetFieldCount()):
        target_field_defn = target_layer_defn.GetFieldDefn(i)
        target_layer.CreateField(target_field_defn)

    # main loop
    for feature1 in layer1:
        # print(feature1.items())
        geometry1 = feature1.geometry()
        layer2.SetSpatialFilter(geometry1)
        total = layer2.GetFeatureCount()
        multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)
        for count, feature2 in enumerate(layer2):
            geometry2 = feature2.geometry()
            multipolygon.AddGeometry(geometry2)
            ogr.TermProgress_nocb((count + 1) / total / 3)
        union = multipolygon.UnionCascaded()
        ogr.TermProgress_nocb(0.67)
        geometry1 = geometry1.Difference(union)
        ogr.TermProgress_nocb(0.99)
        feature1.SetGeometry(geometry1)
        target_layer.CreateFeature(feature1)
        ogr.TermProgress_nocb(1)

    return 0


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    # add arguments here
    parser.add_argument('shape1_path', metavar='SHAPE1')
    parser.add_argument('shape2_path', metavar='SHAPE2')
    parser.add_argument('target_path', metavar='TARGET')
    return parser


def main():
    """ Call command with args from parser. """
    kwargs = vars(get_parser().parse_args())

    logging.basicConfig(stream=sys.stderr,
                        level=logging.DEBUG,
                        format='%(message)s')

    try:
        return command(**kwargs)
    except Exception:
        logger.exception('An exception has occurred.')


if __name__ == '__main__':
    exit(main())
