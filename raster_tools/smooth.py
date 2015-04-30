from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import os
import sys
import datetime
import logging
import subprocess

import numpy as np
from scipy.signal import fftconvolve
from osgeo import gdal
from osgeo import gdal_array
from osgeo import ogr
from osgeo import osr

import unipath

logger = logging.getLogger(__name__)


def get_image(tile_name):
    tile_name_to_match = tile_name[1:]
    lookup_dir = tile_name_to_match[:3]
    tile_path = unipath.Path(AHN2_INT, lookup_dir,
                             'n{0:s}.tif'.format(tile_name_to_match))
    return tile_path


def orfeo_smooth(tile_in, tile_out):
    logger.info('[*] Starting smoothing..')
    # import ipdb; ipdb.set_trace()

    ps = subprocess.Popen(
        ['otbcli_Smoothing', '-in', tile_in,
         '-out', tile_out,
         'float', '-type', 'gaussian', '-type.gaussian.radius', '20'],
        stdout=subprocess.PIPE
    )

    output = ps.communicate()[0]
    for line in output.splitlines():
        logger.debug("[*] {0}".format(line))


def merge(first, second, out_file):
    pass


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

    def gaussian_blur(self, in_array, size):
        # expand in_array to fit edge of kernel
        # import ipdb; ipdb.set_trace()
        padded_array = np.pad(in_array, size, mode='symmetric')
        # build kernel
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        g = np.exp(-(x**2 / float(size) + y**2 / float(size)))
        g = (g / g.sum()).astype(in_array.dtype)
        # do the Gaussian blur
        return fftconvolve(padded_array, g, mode='valid')


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr,
                        level=logging.DEBUG,
                        format='%(message)s')

    logger.info("[*] Kick off!")
    # build path

    # process image 1) smooth 2) merge

    PROJECT_DIR = unipath.Path(os.path.abspath(__file__)).parent.ancestor(1)
    TOOLS_DIR = PROJECT_DIR.child('raster_tools')
    CSV_DIR = PROJECT_DIR.child('csv')
    AHN2_ROOT = unipath.Path('/mnt/Projectendata/Data_Sources/raster-sources/ahn2')
    AHN2_INT = unipath.Path(AHN2_ROOT, 'int')
    tiles_file = '{0:s}/subset_adam.csv'.format(CSV_DIR)
    out_dir = unipath.Path('/home/lars_claussen/Data/ahn2/out_adam')


    t1 = datetime.datetime.now()
    # readin txt
    try:
        with open(tiles_file, 'rb') as f:
            for i,line in enumerate(f):
                # call function
                if i > 0:
                    break
                logger.debug("[DB] Line: {0}".format(line))
                tile_raw = line.strip()
                tile_path = get_image(tile_raw)
                if not os.path.exists(tile_path):
                    logger.warning("[!] File {0:s} does not "
                                   "exist! Skipping...".format(
                        os.path.basename(tile_path)))
                    continue
                out_file = unipath.Path(out_dir, '{0:s}_GGsth{1:s}'.format(
                    tile_path.stem, tile_path.ext))
                logger.debug("tile path {0}, out file {1}".format(tile_path, out_file))
                # orfeo_smooth(tile_path, out_file)
                ri = RasterImage(tile_path)
                my_array = ri.raster2array()
                blurred = ri.gaussian_blur(blurred, 10)
                ri.array2raster(my_array, out_file)
            logger.info('[+] Smoothing successful!')
            logger.info("[*] Execution time: %s" % (datetime.datetime.now() - t1))
    except IOError, ioerr:
        print(ioerr)
        raise




