# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
"""
Create index TARGET that covers at least the features from SOURCE,
but with square units of size SIZE.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import string

from raster_tools import ogr

driver = ogr.GetDriverByName(str('esri shapefile'))


class Checker(object):
    def __init__(self, layer):
        self.layer = layer

    def intersects(self, geometry):
        self.layer.SetSpatialFilter(geometry)
        for feature in self.layer:
            intersection = geometry.Intersection(feature.geometry())
            if intersection.GetGeometryType() == ogr.wkbPolygon:
                return True
        return False


def reindex(source_path, target_path, size):
    source_data_source = ogr.Open(source_path)
    source_layer = source_data_source[0]
    sr = source_layer.GetSpatialRef()
    checker = Checker(source_layer)
    extent = source_layer.GetExtent()

    target_data_source = driver.CreateDataSource(target_path)
    target_layer_name = os.path.basename(target_path)
    target_layer = target_data_source.CreateLayer(target_layer_name, sr)
    target_layer.CreateField(ogr.FieldDefn(str('name'), ogr.OFTString))
    target_layer_defn = target_layer.GetLayerDefn()

    xcorners = range(int(extent[0]), int(extent[2]), size)
    ycorners = range(int(extent[3]), int(extent[1]), -size)
    total = len(ycorners)
    chars = string.ascii_lowercase
    ogr.TermProgress_nocb(0)
    for m, y in enumerate(ycorners):
        for n, x in enumerate(xcorners):
            x1, y1, x2, y2 = x, y - size, x + size, y
            feature = ogr.Feature(target_layer_defn)
            xname = (chars[n // (26 * 26)]
                     + chars[(n // 26) % 26]
                     + chars[n % 26] + '{:04.0f}'.format(n))
            feature[str('name')] = xname
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint_2D(x1, y1)
            ring.AddPoint_2D(x2, y1)
            ring.AddPoint_2D(x2, y2)
            ring.AddPoint_2D(x1, y2)
            ring.AddPoint_2D(x1, y1)
            geometry = ogr.Geometry(ogr.wkbPolygon)
            geometry.AddGeometry(ring)
            if not checker.intersects(geometry):
                continue
            geometry.AssignSpatialReference(sr)
            feature.SetGeometry(geometry)
            target_layer.CreateFeature(feature)
        ogr.TermProgress_nocb((m + 1) / total)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_path', metavar='SOURCE')
    parser.add_argument('target_path', metavar='TARGET')
    parser.add_argument('-s', '--size', type=int, default=4096)
    return parser


def main():
    """ Call command with args from parser. """
    kwargs = vars(get_parser().parse_args())
    reindex(**kwargs)
