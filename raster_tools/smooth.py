# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
"""
Smooth raster images using a low-pass filter (gaussian blur)

Example usage from /Data_Sources/raster-sources/ahn2:

    smooth index ahn2 ahn2/sth --use_mask True --buffer 5 --gaussian 20
"""


from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import os
import sys
import datetime
import logging
import argparse

import numpy as np
from scipy import signal
from scipy import ndimage
import unipath

from osgeo import ogr
from osgeo import gdal

gdal.UseExceptions()

logger = logging.getLogger(__name__)

NO_DATA_VALUE = -3.4028234663852886e+38
GDAL_GTIFF_DRIVER = gdal.GetDriverByName(b'gtiff')


def get_image(tile_name, base_dir, return_lookup_dir=False):
    """get AHN2 raster"""

    tile_name_to_match = tile_name[1:]
    lookup_dir = tile_name_to_match[:3]
    tile_path = unipath.Path(base_dir, lookup_dir,
                             'n{0:s}.tif'.format(tile_name_to_match))
    if return_lookup_dir:
        return tile_path, lookup_dir
    return tile_path


class RasterImage(object):

    def __init__(self, image_name):
        self.image = image_name
        self.raster = None

    def raster2array(self):
        self.raster = gdal.Open(self.image)
        band = self.raster.GetRasterBand(1)
        return band.ReadAsArray()

    def array2raster(self, array, file_name):
        target = GDAL_GTIFF_DRIVER.Create(
            file_name,
            self.raster.RasterXSize,
            self.raster.RasterYSize,
            1,
            gdal.GDT_Float32,
            options=['compress=deflate', 'tiled=yes'])

        target.SetProjection(self.raster.GetProjection())
        target.SetGeoTransform(self.raster.GetGeoTransform())
        target.GetRasterBand(1).WriteArray(array)
        target.FlushCache()  # Write to disk.

    @staticmethod
    def get_nodata_value_mask(non_array, int_array, buffer_size):
        # get a boolean mask
        mask = non_array == NO_DATA_VALUE
        # import ipdb; ipdb.set_trace()
        buffered_mask = RasterImage.boolean_buffer(mask, buffer_size)
        # index the no data cells, fill the rest with 0
        interpolated_nodata = int_array*buffered_mask
        return interpolated_nodata

    @staticmethod
    def boolean_buffer(boolean_array, buffer_size):
        """
        Expands the True area in an array 'input'.

        Expansion occurs in the horizontal and vertical directions by
        ``buffer_size`` cell
        """
        return ndimage.morphology.binary_dilation(boolean_array,
                                                  iterations=buffer_size)

    @staticmethod
    def gaussian_blur(in_array, size):
        # expand in_array to fit edge of kernel
        padded_array = np.pad(in_array, size, mode='symmetric'.encode('ascii'))
        # build kernel
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        g = np.exp(-(x**2 / float(size) + y**2 / float(size)))
        g = (g / g.sum()).astype(in_array.dtype)
        # do the Gaussian blur
        return signal.fftconvolve(padded_array, g, mode='valid')

    @staticmethod
    def merge_arrays(first, second):
        """
        TODO: make where clause variable
        """
        return np.where(second == NO_DATA_VALUE, first, second)


class Config(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument('index_path', metavar='INDEX',
                            help='raster index shape file incl full path')
        parser.add_argument('ahn2_root', metavar='AHN2_ROOT',
                            help='path to the dir where the AHN2 rasters '
                                 'are stored. Should contain folder '
                                 '"int", "non" etc')
        parser.add_argument('target_dir', metavar='TARGET',
                            help='path were the result files should '
                                 'be written to')
        parser.add_argument('-g', '--gaussian',
                            default=25,
                            type=int,
                            help="gaussian blur radius to "
                                 "use (default is 25)")
        parser.add_argument('-m', '--use_mask',
                            type=bool, default=False,
                            help="if True won't smooth the whole tile but "
                                 "only NaN areas. "
                                 "Default is False")
        parser.add_argument('-b', '--buffer',
                            type=int, default=10,
                            help="if a mask is used for the smoothing"
                                 " the buffer option can be used to "
                                 "enlarge the mask by <buffer> cells. "
                                 "This will result in "
                                 "softer transitions between the area of "
                                 "the mask and its surroundings")
        self.args = parser.parse_args()

    def project_layout(self):
        """
        """
        self.project_dir = unipath.Path(
            os.path.abspath(__file__)).parent.ancestor(1)
        self.tools_dir = self.project_dir.child('raster_tools')
        self.ahn2_root = unipath.Path(self.args.ahn2_root)
        self.ahn2_INT = self.ahn2_root.child('int')
        self.ahn2_NON = self.ahn2_root.child('non')
        self.out_dir = unipath.Path(self.args.target_dir)
        self.out_dir.mkdir()


def main():
    logging.basicConfig(stream=sys.stderr,
                        level=logging.INFO,
                        format='%(message)s')

    logger.info("[*] Starting smoothing...")

    conf = Config()
    conf.project_layout()

    data_source = ogr.Open(conf.args.index_path)
    layer = data_source[0]
    total = layer.GetFeatureCount()
    logger.debug("[DB] Counted {0} features".format(total))

    t1 = datetime.datetime.now()
    for count, feature in enumerate(layer, 1):
        # get tile name from index.shp
        tile_raw = feature[b'bladnr']
        logger.debug("[DB] tile_raw: {0}".format(tile_raw))

        # interpolated and not-interpolated AHN2 tiles
        int_tile, sub_dir = get_image(tile_raw, base_dir=conf.ahn2_INT,
                                      return_lookup_dir=True)
        non_tile = get_image(tile_raw, base_dir=conf.ahn2_NON)
        # we need them both, so skip this iteration if one of
        # them is missing
        if not os.path.exists(int_tile) or not os.path.exists(non_tile):
            logger.debug("[!] File {0:s} does not exist! "
                         "Skipping...".format(os.path.basename(int_tile)))
            continue

        ri_int = RasterImage(int_tile)
        int_array = ri_int.raster2array()
        ri_non = RasterImage(non_tile)
        non_array = ri_non.raster2array()

        # result file naming...
        out_sub = conf.out_dir.child(sub_dir)
        out_sub.mkdir()
        out_file_gb = unipath.Path(out_sub,
                                   '{0:s}{1:s}'.format(
                                       int_tile.stem, int_tile.ext))
        logger.debug("tile path {0}, out file {1}".format(
            int_tile, out_file_gb))

        if conf.args.use_mask:
            blur_this = ri_int.get_nodata_value_mask(
                non_array, int_array, buffer_size=conf.args.buffer)
        else:
            blur_this = int_array

        blurred = ri_int.gaussian_blur(blur_this, size=conf.args.gaussian)
        result_array = ri_int.merge_arrays(blurred, non_array)
        ri_int.array2raster(result_array, out_file_gb)
        gdal.TermProgress_nocb(count / total)

    logger.info('[+] Smoothing successful!')
    logger.info("[*] Execution time: %s" % (datetime.datetime.now() - t1))
