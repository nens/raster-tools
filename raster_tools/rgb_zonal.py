# -*- coding: utf-8 -*-
"""
Perform zonal calculation on the r, g and b bands
of aerial imagery datasets. Examples of usage:

    $ rgb-zonal image.tif borders.shp output.shp '(r + g + b).mean()'
    $ rgb-zonal image.tif borders.shp output.shp 'np.median(g)'
"""

import argparse
import logging
import sys

from osgeo import gdal
import numpy as np

from raster_tools import datasources
from raster_tools import groups


logger = logging.getLogger(__name__)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'image_path',
        metavar='IMAGE',
        help='Path to GDAL dataset containing RGB data.',
    )
    parser.add_argument(
        'source_path',
        metavar='SOURCE',
        help='Path to shape with source features.',
    )
    parser.add_argument(
        'target_path',
        metavar='TARGET',
        help='Path to shape with target features.',
    )
    parser.add_argument(
        'calculation',
        metavar='CALCULATION',
        help='Calculation using a combination of r, g and b',
    )
    parser.add_argument(
        '-p', '--part',
        help='Partial processing source, for example "2/3"',
    )
    return parser


def command(image_path, source_path, target_path, calculation, part):
    """ Main """
    source_features = datasources.PartialDataSource(source_path)
    if part is not None:
        source_features = source_features.select(part)

    image = groups.RGBWrapper(gdal.Open(image_path))

    target = datasources.TargetDataSource(
        path=target_path,
        template_path=source_path,
        attributes=['result'],
    )

    for source_feature in source_features:
        geometry = source_feature.geometry()

        # skip large areas
        if geometry.GetArea() > 1000:
            continue

        # retrieve raster data
        try:
            data, mask = image.read(geometry)
        except RuntimeError:
            continue

        # skip incomplete data
        if not mask.any() or not data.any():
            continue

        # debug image
        # tmp = np.zeros_like(data)
        # tmp[mask] = data[mask]
        # from PIL import Image
        # Image.fromarray(tmp.transpose(1, 2, 0)).show()

        # apppend feature to output shapefile, with calculation result
        (r, g, b) = (d[m] for d, m in zip(np.int64(data), mask))
        attributes = source_feature.items()
        attributes['result'] = eval(calculation)
        target.append(geometry=geometry, attributes=attributes)


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    command(**vars(get_parser().parse_args()))
