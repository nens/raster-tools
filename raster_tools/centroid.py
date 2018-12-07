# -*- coding: utf-8 -*-
"""
Add the raster value under the centroid of input geometries to a shapefile.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

from osgeo import gdal_array
from osgeo import gdal
from osgeo import osr

import numpy as np

from raster_tools import datasources

logger = logging.getLogger(__name__)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        'source_path',
        metavar='SOURCE',
        help='Path to shape with source features.',
    )
    parser.add_argument(
        'raster_path',
        metavar='RASTER',
        help='Path to gdal raster.',
    )
    parser.add_argument(
        'target_path',
        metavar='TARGET',
        help='Path to shape with target features.',
    )
    parser.add_argument(
        '-a', '--attribute',
        default='value',
        help='Specify an alternative attribute name instead of "value"',
    )
    parser.add_argument(
        '-p', '--part',
        help='Partial processing source, for example "2/3"',
    )
    return parser


class GeoTransform(tuple):

    def get_centroid_indices(self, geometry):
        """Return (i1, i2), (j1, j2) image coordinates.

        :param geometry: ogr geometry.

        Use the inverse of the transformation matrix to calculate
        image subpixel coordinates from the bounding box of the
        supplied geometry.

        Geometry and geo transform must be with respect to the same
        coordinate reference system.
        """
        # spatial coordinates
        x, y = geometry.Centroid().GetPoints()[0]

        # inverse transformation
        p, a, b, q, c, d = self
        (e, f), (g, h) = np.linalg.inv([(a, b), (c, d)])

        # apply to envelope corners
        u = int(e * (x - p) + f * (y - q))
        v = int(g * (x - p) + h * (y - q))

        return u, v


def command(source_path, raster_path, target_path, attribute, part):
    """ Main """
    # open source datasource
    source = datasources.PartialDataSource(source_path)
    if part is not None:
        source = source.select(part)

    raster = gdal.Open(raster_path)
    geo_transform = GeoTransform(raster.GetGeoTransform())
    sr = osr.SpatialReference(raster.GetProjection())
    band = raster.GetRasterBand(1)
    dtype = gdal_array.flip_code(band.DataType)
    no_data_value = np.array(band.GetNoDataValue(), dtype).item()

    # prepare statistics gathering
    target = datasources.TargetDataSource(
        path=target_path,
        template_path=source_path,
        attributes=[attribute],
    )

    for source_feature in source:
        geometry = source_feature.geometry()
        geometry.TransformTo(sr)
        xoff, yoff = geo_transform.get_centroid_indices(geometry)
        value = band.ReadAsArray(xoff, yoff, 1, 1).item()
        attributes = {attribute: None if value == no_data_value else value}
        target.append(geometry=geometry,
                      attributes=attributes)
    return 0


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return command(**vars(get_parser().parse_args()))
