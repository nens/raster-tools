# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
"""
Smooth raster images using a low-pass filter (gaussian blur)
"""


from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import os
import sys
import datetime
import logging
import subprocess
import argparse
import itertools

import numpy as np
from scipy.signal import fftconvolve
import unipath

try:
    from osgeo import ogr
    from osgeo import gdal
except ImportError, _:
    import ogr
    import gdal

logger = logging.getLogger(__name__)

NO_DATA_VALUE = '-3.4028234663852886e+38'


class RasterImage(object):

    def __init__(self, image_name):
        self.image = image_name
        self.raster = None

    def raster2array(self):
        self.raster = gdal.Open(self.image)
        band = self.raster.GetRasterBand(1)
        return band.ReadAsArray()

    def array2raster(self, array, file_name):
        driver = gdal.GetDriverByName(b'gtiff')
        target = driver.Create(
            file_name,
            self.raster.RasterXSize,
            self.raster.RasterYSize,
            1,
            gdal.GDT_Float32)

        target.SetProjection(self.raster.GetProjection())
        target.SetGeoTransform(self.raster.GetGeoTransform())
        _ = target.GetRasterBand(1).WriteArray(array)
        target.FlushCache()  # Write to disk.

    @staticmethod
    def get_nodata_value_mask(non_array, int_array):
        # get a boolean mask
        mask =  non_array == float(NO_DATA_VALUE)
        buffered_mask = RasterImage.boolean_buffer(mask, 3)
        # index the no data cells, fill the rest with 0
        interpolated_nodata = int_array*buffered_mask
        # # if only 0 return int_array
        # if not interpolated_nodata.any():
        #     return int_array
        return interpolated_nodata

    @staticmethod
    def boolean_buffer(boolean_array, iters):
        """
        Expands the True area in an array 'input'.

        Expansion occurs in the horizontal and vertical directions by one
        cell, and is repeated 'iters' times.
        """
        # TODO: this is very slow, probably there is some native numpy function
        # TODO: that can buffer an existing mask?
        y_len,x_len = boolean_array.shape
        output = boolean_array.copy()
        for iter in xrange(iters):
            for y in xrange(y_len):
                for x in xrange(x_len):
                    if (y > 0 and boolean_array[y-1,x]) or \
                            (y < y_len - 1 and boolean_array[y+1,x]) or \
                            (x > 0 and boolean_array[y,x-1]) or \
                            (x < x_len - 1 and boolean_array[y,x+1]):
                        output[y,x] = True
            boolean_array = output.copy()
        return output

    @staticmethod
    def get_image(tile_name, base_dir):
        """get AHN2 raster"""

        tile_name_to_match = tile_name[1:]
        lookup_dir = tile_name_to_match[:3]
        tile_path = unipath.Path(base_dir, lookup_dir,
                                 'n{0:s}.tif'.format(tile_name_to_match))
        return tile_path

    @staticmethod
    def gaussian_blur(in_array, size):
        # expand in_array to fit edge of kernel
        padded_array = np.pad(in_array, size, mode='symmetric'.encode('ascii'))
        # build kernel
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        g = np.exp(-(x**2 / float(size) + y**2 / float(size)))
        g = (g / g.sum()).astype(in_array.dtype)
        # do the Gaussian blur
        return fftconvolve(padded_array, g, mode='valid')

    @staticmethod
    def merge(first, second, out_file):
        """
        This utility will automatically mosaic a set of images.
        All the images must be in the same coordinate system and
        have a matching number of bands, but they may be overlapping,
        and at different resolutions. In areas of overlap,
        the last image will be copied over earlier ones.

        :param first:
        :param second:
        :param out_file:
        :return:
        """
        ps = subprocess.Popen(
            ['gdal_merge.py', '-o', out_file,
             '-of', 'GTiff',
             '-n', NO_DATA_VALUE,
             first, second],
            stdout=subprocess.PIPE
        )
        output = ps.communicate()[0]
        for line in output.splitlines():
            logger.debug("[*] {0}".format(line))

    @staticmethod
    def orfeo_smooth(tile_in, tile_out, smooth_factor=20):
        ps = subprocess.Popen(
            ['otbcli_Smoothing', '-in', tile_in,
             '-out', tile_out,
             'float', '-type', 'gaussian', '-type.gaussian.radius', smooth_factor],
            stdout=subprocess.PIPE
        )

        output = ps.communicate()[0]
        for line in output.splitlines():
            logger.debug("[*] {0}".format(line))


class Config(object):

    def __init__(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('index_path', metavar='INDEX',
                            help='raster index shape file incl full path')
        parser.add_argument('ahn2_root', metavar='AHN2_MAP',
                            help='path to the dir where the AHN2 rasters '
                                 'are stored. Should contain folder '
                                 '"int", "non" etc')
        parser.add_argument('target_dir', metavar='TARGET',
                            help='path were the result files should '
                                 'be written to')
        parser.add_argument('blur', metavar='BLUR',
                            nargs='?', default='25',
                            help="gaussian blur radius to use")
        self.args = parser.parse_args()

    def get_project_layout(self):
        """
        """
        self.project_dir = unipath.Path(os.path.abspath(__file__)).parent.ancestor(1)
        self.tools_dir = self.project_dir.child('raster_tools')
        self.ahn2_root = unipath.Path(self.args.ahn2_root)
        self.ahn2_INT = self.ahn2_root.child('int')
        self.ahn2_NON = self.ahn2_root.child('non')
        self.out_dir = unipath.Path(self.args.target_dir)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr,
                        level=logging.DEBUG,
                        format='%(message)s')

    logger.info("[*] Kick off!")

    conf = Config()
    conf.get_project_layout()

    data_source = ogr.Open(conf.args.index_path)
    layer = data_source[0]
    total = layer.GetFeatureCount()
    logger.debug("[DB] Counted {0} features".format(total))

    t1 = datetime.datetime.now()
    # readin txt
    for count, feature in enumerate(layer, 1):
        if count > 600:
            break
        tile_raw = feature.GetFieldAsString("BLADNR".encode('ascii')).strip()
        logger.debug("[DB] tile_raw: {0}".format(tile_raw))
        int_tile = RasterImage.get_image(tile_raw, base_dir=conf.ahn2_INT)
        non_tile = RasterImage.get_image(tile_raw, base_dir=conf.ahn2_NON)
        if not os.path.exists(int_tile) or not os.path.exists(non_tile):
            logger.warning("[!] File {0:s} does not "
                           "exist! Skipping...".format(
                os.path.basename(int_tile)))
            continue
        out_file_gb = unipath.Path(conf.out_dir, '{0:s}_gb{1:s}'.format(
            int_tile.stem, int_tile.ext))
        logger.debug("tile path {0}, out file {1}".format(int_tile, out_file_gb))
        out_file = unipath.Path(conf.out_dir, '{0:s}_mrg{1:s}'.format(
            int_tile.stem, int_tile.ext))
        # orfeo_smooth(tile_path, out_file)
        ri_int = RasterImage(int_tile)
        ri_non = RasterImage(non_tile)

        int_array = ri_int.raster2array()
        non_array = ri_non.raster2array()

        selected = ri_int.get_nodata_value_mask(non_array, int_array)

        blurred = ri_int.gaussian_blur(selected, 25)
        ri_int.array2raster(blurred, out_file_gb)
        ri_int.merge(out_file_gb, non_tile, out_file)
        gdal.TermProgress_nocb(count / total)

    logger.info('[+] Smoothing successful!')
    logger.info("[*] Execution time: %s" % (datetime.datetime.now() - t1))




