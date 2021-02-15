# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.
"""
Calculate shadows.
"""

import argparse
import itertools
import logging
import math
import os
import sys

from osgeo import gdal
from scipy import ndimage
import numpy as np

from raster_tools import datasets
from raster_tools import datasources
from raster_tools import groups

logger = logging.getLogger(__name__)

driver = gdal.GetDriverByName('gtiff')


class Shadower(object):
    """
    Azimuth and elevation currently taken from
    http://www.nrel.gov/midc/solpos/spa.html:
        http://www.nrel.gov/midc/apps/spa.pl?
        syear=2015&smonth=6&sday=21&eyear=2015&emonth=6&eday=21&step=60
        &stepunit=1&latitude=52.155201&longitude=5.3850453&timezone=2&elev=0
        &press=1000&temp=10&dut1=0.0&deltat=64.797&azmrot=180&slope=0
        &refract=0.5667&field=0&field=1&zip=0
    """
    def __init__(self, raster_path, output_path):
        # see class.__doc__
        azimuth = 216
        elevation = 57

        # put the input raster(s) in a group
        if os.path.isdir(raster_path):
            datasets = [gdal.Open(os.path.join(raster_path, path))
                        for path in sorted(os.listdir(raster_path))]
        else:
            datasets = [gdal.Open(raster_path)]
        self.group = groups.Group(*datasets)

        slope = math.tan(math.radians(elevation))
        m_per_px = self.group.geo_transform[1]

        dx = math.sin(math.radians(azimuth))
        dy = -math.cos(math.radians(azimuth))

        # calculate shift and corresponding elevation change
        self.ds = 1 / max(abs(dx), abs(dy))                 # pixels
        self.dz = self.ds * slope * m_per_px      # meters
        self.dx = dx * self.ds                    # pixels
        self.dy = dy * self.ds                    # pixels

        # calculate margin for input data
        self.mz = 367  # gerbrandy tower, in meters
        ms = self.mz / slope / m_per_px                            # pixels
        self.mx = int(math.copysign(math.ceil(abs(dx * ms)), dx))  # pixels
        self.my = int(math.copysign(math.ceil(abs(dy * ms)), dy))  # pixels

        self.output_path = output_path

    def get_size_and_bounds(self, geometry):
        """
        Return the window into the source raster that includes the
        required margin, and the slices that return from that window
        the part corresponding to geometry.
        """
        x1, y1, x2, y2 = self.group.geo_transform.get_indices(geometry)
        size = x2 - x1, y2 - y1
        bounds = (
            min(x1, x1 + self.mx),
            min(y1, y1 + self.my),
            max(x2, x2 + self.mx),
            max(y2, y2 + self.my),
        )
        return size, bounds

    def get_view(self, array, size, iteration=0):
        """ Return shifted view on array """
        w1, h1 = size
        h2, w2 = array.shape

        dx = int(round(iteration * self.dx))
        dy = int(round(iteration * self.dy))

        if self.mx > 0:
            slice_x = slice(dx, w1 + dx)
        else:
            slice_x = slice(w2 - w1 + dx, w2 + dx)

        if self.my > 0:
            slice_y = slice(dy, h1 + dy)
        else:
            slice_y = slice(h2 - h1 + dy, h2 + dy)

        slices = slice_y, slice_x

        if iteration:
            return array[slices] - iteration * self.dz
        return array[slices]

    def shadow(self, feature):
        # target path
        leaf_number = feature['BLADNR']
        path = os.path.join(self.output_path,
                            leaf_number[:3],
                            '{}.tif'.format(leaf_number))
        if os.path.exists(path):
            logger.debug('Target already exists.')
            return

        # prepare
        geometry = feature.geometry()
        size, bounds = self.get_size_and_bounds(geometry)
        array = self.group.read(bounds)

        # maximum filter makes shadows a little wider
        footprint = ndimage.generate_binary_structure(2, 1)
        array = ndimage.maximum_filter(array, footprint=footprint)

        view1 = self.get_view(array=array, size=size)
        target = np.zeros_like(view1, dtype='b1')

        # calculate shadow
        for iteration in itertools.count(1):
            view2 = self.get_view(array=array, size=size, iteration=iteration)
            index = np.logical_and(~target, view2 > view1)
            if not index.any():
                break
            if iteration * self.dz > self.mz:
                break
            target[index] = True
        target = target.astype('u1') + 255  # True: 0, False: 255

        # create directory
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass  # no problem

        kwargs = {
            'no_data_value': 255,
            'projection': self.group.projection,
            'geo_transform': self.group.geo_transform.shifted(geometry),
        }
        options = [
            'tiled=yes',
            'compress=deflate',
        ]
        with datasets.Dataset(target[np.newaxis], **kwargs) as dataset:
            driver.CreateCopy(path, dataset, options=options)


def shadow(index_path, raster_path, output_path, part):
    """
    """
    index = datasources.PartialDataSource(index_path)
    if part is not None:
        index = index.select(part)

    shadower = Shadower(output_path=output_path,
                        raster_path=raster_path)

    for feature in index:
        shadower.shadow(feature)
    return 0


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        'index_path',
        metavar='INDEX',
        help='shapefile with geometries and names of output tiles',
    )
    parser.add_argument(
        'raster_path',
        metavar='RASTER',
        help='source GDAL raster or directory GDAL rasters to stack.'
    )
    parser.add_argument(
        'output_path',
        metavar='OUTPUT',
        help='target folder',
    )
    parser.add_argument(
        '-p', '--part',
        help='partial processing source, for example "2/3"',
    )
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


def main():
    """ Call shadow with args from parser. """
    # logging
    kwargs = vars(get_parser().parse_args())
    if kwargs.pop('verbose'):
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level, format='%(message)s')

    # run or fail
    try:
        shadow(**kwargs)
        return 0
    except Exception:
        logger.exception('An exception has occurred.')
        return 1
