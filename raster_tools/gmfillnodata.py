# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import os
import sys
import tempfile

from osgeo import gdal
# from osgeo import gdal_array

import numpy as np

from raster_tools import datasets
from raster_tools import datasources
from raster_tools import groups

DRIVER = gdal.GetDriverByName(str('gtiff'))
OPTIONS = ['compress=deflate', 'tiled=yes']

logger = logging.getLogger(__name__)


class GdalFiller(object):

    def __init__(self, raster_path, output_path):

        if os.path.isdir(raster_path):
            raster_datasets = [gdal.Open(os.path.join(raster_path, path))
                               for path in sorted(os.listdir(raster_path))]
        else:
            raster_datasets = [gdal.Open(raster_path)]

        self.raster_group = groups.Group(*raster_datasets)
        self.output_path = output_path

        # properties
        self.projection = self.raster_group.projection
        self.geo_transform = self.raster_group.geo_transform
        self.no_data_value = self.raster_group.no_data_value.item()

    def fill(self, feature):
        """
        Call gdal interpolation function
        """
        # prepare target path
        name = feature[str('bladnr')]
        path = os.path.join(self.output_path,
                            name[:3],
                            '{}.tif'.format(name))
        if os.path.exists(path):
            return

        # create directory
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass  # no problem

        # check for data
        geometry = feature.geometry()
        values = self.raster_group.read(geometry)
        if (values == self.no_data_value).all():
            return

        kwargs = {
            'projection': self.projection,
            'geo_transform': self.geo_transform.shifted(geometry),
            'no_data_value': self.no_data_value,
        }

        # gdal is going to use the current dir as temporary space
        curdir = os.getcwd()
        tmpdir = tempfile.mkdtemp(dir='/dev/shm')
        os.chdir(tmpdir)

        # fill no data until no voids remain
        iterations = 0
        original_values = values.copy()  # for diffing
        while self.no_data_value in values:

            # create a mask band
            # mask_array = (values != self.no_data_value).view('u1')
            # mask = datasets.create(mask_array[np.newaxis])
            # mask_band = mask.GetRasterBand(1)

            # call the algorithm
            with datasets.Dataset(values[np.newaxis], **kwargs) as work:
                work_band = work.GetRasterBand(1)
                mask_band = work_band.GetMaskBand()
                try:
                    gdal.FillNodata(
                        work_band,
                        mask_band,
                        100,  # search distance
                        0,    # smoothing iterations
                    )
                except RuntimeError:
                    print(name)
                    raise
            iterations += 1

        # switch back current dir
        os.chdir(curdir)
        os.rmdir(tmpdir)

        # write diff
        values[values == original_values] = self.no_data_value
        with datasets.Dataset(values[np.newaxis], **kwargs) as result:
            DRIVER.CreateCopy(path, result, options=OPTIONS)


def gmfillnodata(index_path, part, **kwargs):
    """
    Fill no data using the gdal algorithm.
    """
    # select some or all polygons
    index = datasources.PartialDataSource(index_path)
    if part is not None:
        index = index.select(part)

    gdal_filler = GdalFiller(**kwargs)

    for feature in index:
        gdal_filler.fill(feature)
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
        help='source GDAL raster with voids.'
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
    return parser


def main():
    """ Call gmfillnodata with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return gmfillnodata(**vars(get_parser().parse_args()))
