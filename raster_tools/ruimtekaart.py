# -*- coding: utf-8 -*-
"""
Compute the sum of 12 rasterfiles in polygons given by a shapefile and compute
the "ruimte-indicator". Optionally, a mask shapefile can be provided.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from osgeo import gdal, ogr, gdal_array

import argparse
import logging
import os

import numpy as np
from scipy import ndimage

from .datasets import Dataset

logger = logging.getLogger(__name__)


DAMAGE_TIFF_NAMES = (
    'gg_t10.tif',
    'ghg_t10.tif',
    'gg_t100.tif',
    'ghg_t100.tif',
    'gg_t1000.tif',
    'ghg_t1000.tif',
)
MAXDEPTH_TIFF_NAMES = DAMAGE_TIFF_NAMES

ATTRS = {
    'damage_0': 'dggt10',
    'damage_1': 'dght10',
    'damage_2': 'dggt100',
    'damage_3': 'dght100',
    'damage_4': 'dggt1000',
    'damage_5': 'dght1000',
    'volume_0': 'mggt10',
    'volume_1': 'mght10',
    'volume_2': 'mggt100',
    'volume_3': 'mght100',
    'volume_4': 'mggt1000',
    'volume_5': 'mght1000',
    'deuro_per_dm3_0': 'ekght10',
    'deuro_per_dm3_1': 'ekggt100',
    'deuro_per_dm3_2': 'ekght100',
    'deuro_per_dm3_3': 'ekggt1000',
    'deuro_per_dm3_4': 'ekght1000',
    'deuro_per_dm3_norm_0': 'nekght10',
    'deuro_per_dm3_norm_1': 'nekggt100',
    'deuro_per_dm3_norm_2': 'nekght100',
    'deuro_per_dm3_norm_3': 'nekggt1000',
    'deuro_per_dm3_norm_4': 'nekght1000',
    'indicator': 'indicator',
}
MIN_BLOCKSIZE = 1024  # process rasters using at least 1024x1024 blocks


def _iter_block_row(band, offset_y, block_height, block_width, no_data_value):
    ncols = int(band.XSize / block_width)
    for i in range(ncols):
        arr = band.ReadAsArray(i * block_width, offset_y,
                               block_width, block_height)
        if no_data_value is not None:
            arr[arr == no_data_value] = 0.
        yield (i * block_width, offset_y,
               (i + 1) * block_width, offset_y + block_height), arr

    # possible leftover block
    width = band.XSize - (ncols * block_width)
    if width > 0:
        arr = band.ReadAsArray(i * block_width, offset_y,
                               width, block_height)
        if no_data_value is not None:
            arr[arr == no_data_value] = 0.
        yield (ncols * block_width, offset_y,
               ncols * block_width + width, offset_y + block_height), arr


def iter_blocks(band, min_blocksize=0):
    """ Iterate over native blocks in a GDal raster data band.

    Optionally, provide a minimum block dimension.

    Returns a tuple of bbox (x1, y1, x2, y2) and the data as ndarray. """
    block_height, block_width = band.GetBlockSize()
    block_height = max(min_blocksize, block_height)
    block_width = max(min_blocksize, block_width)

    nrows = int(band.YSize / block_height)
    no_data_value = band.GetNoDataValue()
    for j in range(nrows):
        for block in _iter_block_row(band, j * block_height, block_height,
                                     block_width, no_data_value):
            yield block

    # possible leftover row
    height = band.YSize - (nrows * block_height)
    if height > 0:
        for block in _iter_block_row(band, nrows * block_height,
                                     height, block_width,
                                     no_data_value):
            yield block


def analyze_raster(filepath):
    """ Analyze a rasterfile to get projection, geo_transform, and shape """
    raster = gdal.Open(filepath, 0)
    band = raster.GetRasterBand(1)
    projection = raster.GetProjection()
    geo_transform = raster.GetGeoTransform()
    shape = band.YSize, band.XSize
    area_per_px = abs(geo_transform[1] * geo_transform[5])
    del raster
    return projection, geo_transform, shape, area_per_px


def analyze_shapefile(filepath):
    """ Analyze the shapefile """
    shp_in = ogr.Open(filepath, 0)
    if shp_in is None:
        raise IOError("The shapefile '{}' is invalid.".format(filepath))
    layer_in = shp_in.GetLayer(0)
    if layer_in.GetGeomType() != ogr.wkbPolygon:
        raise ValueError("The shapefile does not contain features of type "
                         "'wkbPolygon'")
    num_regions = layer_in.GetFeatureCount()
    if num_regions > 254:
        raise ValueError("Too many regions ({}), maximum is "
                         "{}".format(num_regions, 254))
    region_ids = []
    for i in range(num_regions):
        feature = layer_in.GetFeature(i)
        region_ids.append(int(feature[str('id')]))
    del shp_in
    return num_regions, region_ids


def analyze_maskfile(filepath):
    """ Analyze the maskfile """
    mask_in = ogr.Open(filepath, 0)
    if mask_in is None:
        raise IOError("The maskfile '{}' is invalid.".format(filepath))
    mask_layer = mask_in.GetLayer(0)
    if mask_layer.GetGeomType() != ogr.wkbPolygon:
        raise ValueError("The maskfile does not contain features of type "
                         "'wkbPolygon'")
    del mask_in


def make_result(filepath_in, filepath_out):
    # copy the shapefile
    shp_in = ogr.Open(filepath_in, 0)
    layer_in = shp_in.GetLayer(0)
    shp_driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.isfile(filepath_out):
        os.remove(filepath_out)
    shp_out = shp_driver.CreateDataSource(filepath_out)
    layer_out = shp_out.CreateLayer(layer_in.GetName(), layer_in.GetSpatialRef(), ogr.wkbPolygon)
    layer_definition = layer_in.GetLayerDefn()
    for i in range(layer_definition.GetFieldCount()):
        layer_out.CreateField(layer_definition.GetFieldDefn(i))

    layer_out.CreateField(ogr.FieldDefn(str('rmtk_id'), ogr.OFTInteger))
    # define extra fields for later use
    for key, name in ATTRS.items():
        layer_out.CreateField(ogr.FieldDefn(str(name), ogr.OFTReal))

    for i in range(layer_in.GetFeatureCount()):
        feature = layer_in.GetFeature(i)
        layer_out.CreateFeature(feature)

    for i in range(layer_out.GetFeatureCount()):
        feature = layer_out.GetFeature(i)
        feature[str('RMTK_ID')] = i
        layer_out.SetFeature(feature)

    # close the input file
    del shp_in

    return shp_out


def rasterize(shape_file, projection, geo_transform, shape,
              mask_path=None):
    """ Rasterize the shapefile into a raster with region ids. """
    if (shape[1] * shape[0]) * 2 > 100 * 1024 * 1024:
        logger.warning("In-memory raster of shape {}x{}".format(*shape))

    layer = shape_file.GetLayer(0)
    no_data_value = np.iinfo(np.uint16).max
    labels = np.full(shape, no_data_value, dtype=np.uint16)[np.newaxis]

    with Dataset(labels, geo_transform=geo_transform,
                 projection=projection,
                 no_data_value=no_data_value) as dataset:
        gdal.RasterizeLayer(dataset, (1,), layer,
                            options=['ATTRIBUTE=RMTK_ID'])

    labels = labels[0]

    if mask_path:
        # rasterize the maskfile
        mask_in = ogr.Open(os.path.join(DATA_ROOT, mask_filename), 0)
        mask_layer = mask_in.GetLayer(0)

        mask = np.zeros(shape, dtype=np.uint8)[np.newaxis]

        with Dataset(mask, geo_transform=geo_transform,
                     projection=projection) as dataset:
            gdal.RasterizeLayer(dataset, (1,), mask_layer, burn_values=(1,))
        mask = mask[0].astype(np.bool)
        del mask_in

    # mask the rasterized shapefile
    labels[~mask] = no_data_value
    return labels


def aggregate(raster_path, out, mask_path, num_regions,
              min_blocksize=1024):
    projection, geo_transform, shape, area_per_px = analyze_raster(raster_path)
    labels = rasterize(out, projection, geo_transform, shape, mask_path)

    source = gdal.Open(raster_path, 0)
    band = source.GetRasterBand(1)

    accum = None
    index = np.arange(num_regions)
    for bbox, block in iter_blocks(band, min_blocksize):
        result = ndimage.sum(block,
                             labels=labels[bbox[1]:bbox[3], bbox[0]:bbox[2]],
                             index=index)
        if accum is None:
            accum = result
        else:
            accum += result

    del source
    return accum, area_per_px


def command(maxdepth_dir, damage_dir, shapefile_path, output_dir,
            mask_path=''):
    num_regions, region_ids = analyze_shapefile(shapefile_path)
    if mask_path:
        analyze_maskfile(mask_path)
    if os.path.isdir(output_dir):
        raise IOError("The output directory '{}' should not exist "
                      "already.".format(output_dir))
    for fn in MAXDEPTH_TIFF_NAMES:
        filepath = os.path.join(maxdepth_dir, fn)
        if not os.path.isfile(filepath):
            raise IOError("'{}' does not exist".format(filepath))
    for fn in DAMAGE_TIFF_NAMES:
        filepath = os.path.join(damage_dir, fn)
        if not os.path.isfile(filepath):
            raise IOError("'{}' does not exist".format(filepath))

    out_filepath = os.path.join(output_dir, 'ruimtekaart.shp')
    out = make_result(shapefile_path, out_filepath)

    # aggregate the 12 input rasters
    damages_euro = np.zeros((num_regions, 6), dtype=np.float)
    volumes_m3 = np.zeros((num_regions, 6), dtype=np.float)

    for i, fn in enumerate(MAXDEPTH_TIFF_NAMES):
        filepath = os.path.join(maxdepth_dir, fn)
        logger.info("Aggregating '{}'".format(filepath))
        volumes_m3[:, i], area_per_px = aggregate(filepath, out, mask_path,
                                                  num_regions, MIN_BLOCKSIZE)
        volumes_m3[:, i] *= area_per_px

    for i, fn in enumerate(DAMAGE_TIFF_NAMES):
        filepath = os.path.join(damage_dir, fn)
        logger.info("Aggregating '{}'".format(filepath))
        damages_euro[:, i], _ = aggregate(filepath, out, mask_path,
                                          num_regions, MIN_BLOCKSIZE)

    # add the sums
    m3 = np.concatenate([volumes_m3,
                         np.sum(volumes_m3, axis=0)[np.newaxis]], axis=0)
    euro = np.concatenate([damages_euro,
                           np.sum(damages_euro, axis=0)[np.newaxis]], axis=0)

    # take the forward differential
    d_euro = np.diff(euro, axis=1)
    d_m3 = np.diff(m3, axis=1)

    # how many euros extra per m3 extra?
    mask = (d_m3 != 0).all(axis=1)
    d_euro_per_m3 = np.zeros_like(d_m3)
    d_euro_per_m3[mask] = d_euro[mask] / d_m3[mask]

    # normalize on the total sum
    d_euro_per_m3_norm = \
        (d_euro_per_m3[-1] - d_euro_per_m3) / d_euro_per_m3[-1]
    d_euro_per_m3_norm[d_euro_per_m3_norm > 1] = 1

    indicator = (d_euro_per_m3_norm[:, 0] +
                 d_euro_per_m3_norm[:, 1] * 2 +
                 d_euro_per_m3_norm[:, 2] * 2 +
                 d_euro_per_m3_norm[:, 3] * 5 +
                 d_euro_per_m3_norm[:, 4] * 5) / 15

    # add the results to the output file
    layer_out = out.GetLayer(0)
    for i in range(layer_out.GetFeatureCount()):
        feature = layer_out.GetFeature(i)

        assert feature['id'] == region_ids[i]

        # set the total damages
        for j in range(6):
            feature[ATTRS['damage_{}'.format(j)]] = euro[i, j]

        # set the total volumes
        for j in range(6):
            feature[ATTRS['volume_{}'.format(j)]] = m3[i, j]

        # set the incremental euro/m3
        for j in range(5):
            feature[ATTRS['deuro_per_dm3_{}'.format(j)]] = d_euro_per_m3[
                i, j]

        # set the normalized incremental euro/m3
        for j in range(5):
            feature[ATTRS['deuro_per_dm3_norm{}'.format(j)]] = \
            d_euro_per_m3_norm[i, j]

        # set the indicator
        if mask[i]:
            feature[ATTRS['indicator']] = indicator[i]
        layer_out.SetFeature(feature)

    # close the output file
    del out


def get_parser():
    """
    Compute the sum of 12 rasterfiles in region given by polygons in a
    shapefile. The 12 raster files are suffixed by shapefile and compute
    the "ruimte-indicator". Optionally, a mask shapefile can be provided.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'maxdepth_dir',
        help=('Path to directory that contains 6 maxdepth tiffiles (in meters)'
              'with the filenames ' + ', '.join(MAXDEPTH_TIFF_NAMES)),
    )
    parser.add_argument(
        'damage_dir',
        help=('Path to directory that contains 6 damage tiffiles (in euros)'
              'with the filenames ' + ', '.join(DAMAGE_TIFF_NAMES)),
    )
    parser.add_argument(
        'shapefile_path',
        help=('Path to a single shapefile that contains the region polygons'),
    )
    parser.add_argument(
        'output_dir',
        help=('Directory name to use as output path'),
    )
    parser.add_argument(
        '-m', '--mask_path',
        default='',
        help=('Path to a mask shapefile containing polygons'),
    )
    return parser


def main():
    """ Call command with args from parser. """
    command(**vars(get_parser().parse_args()))
