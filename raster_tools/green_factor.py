# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.
"""
Calculates greenfactor of polygons (e.g. gardens) with
input: aerial image and polygons and
output: greenfactor for every polygon.

The script is made by Nelen & Schuurmans - Arnold van 't Veld.
"""

import argparse
import subprocess
import os
import re

import numpy as np

from raster_tools import gdal
from raster_tools import ogr
from raster_tools import osr

from raster_tools import datasets
from raster_tools import datasources

DRIVER_OGR_MEMORY = ogr.GetDriverByName('Memory')
DRIVER_GDAL_MEM = gdal.GetDriverByName('mem')
DRIVER_GDAL_GTIFF = gdal.GetDriverByName('gtiff')

WKT_RD = osr.GetUserInputAsWKT('epsg:28992')
PROJ4_RD = osr.SpatialReference(WKT_RD).ExportToProj4().strip()


def command(gardens_path, aerial_image_path,
            target_path, min_green, max_green, check_rast, part, pk):
    # open input shapefile
    shape_gardens = ogr.Open(gardens_path)
    layer_gardens = shape_gardens[0]

    # check projection of input shapefile
    sr = layer_gardens.GetSpatialRef()
    check_sr = get_projection(sr)
    if check_sr is None:
        print('[!] ERROR : EPSG projection code missing from shape.')
        hint = '[*] INFO  : Use this command: gdalsrsinfo -o wkt EPSG:28992 > '
        print(hint, gardens_path.replace('.shp', '.prj'))
        return 1

    # check spaces in aerial_image_path
    if ' ' in aerial_image_path:
        hint = '[!] ERROR : There is a space in the aerial image path: '
        print(hint, aerial_image_path)
        return 1

    # check if raster exists
    if not os.path.isfile(aerial_image_path):
        hint = '[!] ERROR : The specified file could not be found: '
        print(hint, aerial_image_path)
        return 1

    # get raster info and check rasterformat
    rasterfmt = os.path.splitext(aerial_image_path)[1]
    if (rasterfmt == '.ecw' or rasterfmt == '.tif' or rasterfmt == '.vrt'):
        pixelsize, envelope_aerial_image = gdalinfo(aerial_image_path)
    else:
        hint = '[!] ERROR : No supported raster format selected: '
        print(hint, rasterfmt, ' (use .tif, .vrt or .ecw)')
        return 1

    # delete any existing target
    driver = ogr.GetDriverByName('ESRI Shapefile')
    try:
        driver.DeleteDataSource(target_path)
    except RuntimeError:
        pass

    # prepare target dataset
    target_shape = driver.CreateDataSource(target_path)
    target_layer_name = os.path.basename(target_path)
    target_layer = target_shape.CreateLayer(target_layer_name, sr)
    wkt = osr.GetUserInputAsWKT('EPSG:28992')
    open(target_path.replace('.shp', '.prj'), 'w').write(wkt)
    target_layer_defn = layer_gardens.GetLayerDefn()
    for i in range(target_layer_defn.GetFieldCount()):
        target_field_defn = target_layer_defn.GetFieldDefn(i)
        target_layer.CreateField(target_field_defn)
    target_field_defn = ogr.FieldDefn('green_perc', ogr.OFTReal)
    target_field_defn.SetWidth(5)
    target_field_defn.SetPrecision(3)
    target_layer.CreateField(target_field_defn)
    target_field_defn = ogr.FieldDefn('green_area', ogr.OFTReal)
    target_field_defn.SetWidth(10)
    target_field_defn.SetPrecision(2)
    target_layer.CreateField(target_field_defn)

    # create tmp_dir for files
    tmp_dirname = os.path.join(os.path.dirname(target_path), 'tmp_greenfactor')
    if not os.path.exists(tmp_dirname):
        os.makedirs(tmp_dirname)

    # initialise progress bar
    total = layer_gardens.GetFeatureCount()
    ogr.TermProgress_nocb(0)

    index = datasources.PartialDataSource(gardens_path)
    if part is not None:
        index = index.select(part)

    # main loop
    for count, feature_garden in enumerate(index):
        # get envelope and convert it to the projwin for ecw_gdal_translate
        geometry_garden = feature_garden.geometry()
        garden_name = feature_garden.GetFieldAsString(str(pk))
        envelope_garden = geometry_garden.GetEnvelope()
        skip_large_size = check_envelopes_input(garden_name,
                                                envelope_garden,
                                                envelope_aerial_image)
        if not skip_large_size == 1:
            x1, x2, y1, y2 = envelope_garden
            x1, y1 = np.floor(np.divide((x1, y1), pixelsize)) * pixelsize
            x2, y2 = np.ceil(np.divide((x2, y2), pixelsize)) * pixelsize
            envelope_garden_round = (x1, x2, y1, y2)
            projwin_garden = '%s %s %s %s' % (x1, y2, x2, y1)

            # create filename clipped aerial image geotiff
            tmp_aerial_tif = os.path.join(tmp_dirname, garden_name + '.tif')
            tmp_mask_tif = os.path.join(tmp_dirname, garden_name + '_mask.tif')
            tmp_green_tif = os.path.join(tmp_dirname,
                                         garden_name + '_green.tif')

            # prepare rgbt-array for green factor calculation
            rgbt, rasterdata = prepare_aerial_array(tmp_aerial_tif,
                                                    aerial_image_path,
                                                    projwin_garden, sr)

            # prepare mask-array for green factor calculation
            z = rgbt[:1, :, :] * 0
            rasterdata['fillvalue'] = 1
            array = create_filled_rasterarray(z, geometry_garden,
                                              tmp_mask_tif,
                                              rasterdata, print_rast=0)
            m = array[0].astype('b1')

            # Call function to determine the green factor
            result = determine_green_factor(rgbt, m, min_green, max_green)
            greenfactor, greengarden_percentage, greengarden_pixels = result
            greengarden_area = greengarden_pixels * pixelsize ** 2

            # Create raster to check green factor
            if check_rast == 1:
                rasterdata['fillvalue'] = rasterdata['no_data_value'] = -9999
                polygon_envelope_garden = get_envelope_polygon(
                    envelope_garden_round, rasterdata['sr'],
                )
                outside = polygon_envelope_garden.Difference(geometry_garden)
                create_filled_rasterarray(greenfactor[np.newaxis],
                                          outside, tmp_green_tif,
                                          rasterdata, print_rast=1)

            # Create a new feature in shapefile
            attributes_garden = feature_garden.items()
            feature = ogr.Feature(target_layer.GetLayerDefn())
            feature.SetGeometry(geometry_garden)
            for key, value in attributes_garden.items():
                feature[key] = value
            feature['green_perc'] = greengarden_percentage
            feature['green_area'] = greengarden_area
            target_layer.CreateFeature(feature)

            # remove raster
            os.remove(tmp_aerial_tif)

            # progress bar
            ogr.TermProgress_nocb((count + 1) / total)
    return 0


def determine_green_factor(rgbt, m, min_green, max_green):

    fullgarden = rgbt[:, m].astype('f4')
    garden_size = max(fullgarden.shape[1], 1)
    greenfactor = (
        np.sum([rgbt[0], rgbt[2]], axis=0)
        / (np.sum([rgbt[1], rgbt[1]], axis=0) + 0.0001)
    ).astype('f4')
    green_mask = np.array(
        (greenfactor > (np.zeros(rgbt[0].shape) + min_green))
        & (greenfactor < (np.zeros(rgbt[0].shape) + max_green))
    )
    greengarden_mask = np.logical_and(green_mask, m).astype('b1')
    greengarden = rgbt[:, greengarden_mask]
    greengarden_percentage = greengarden.shape[1] / garden_size
    greengarden_pixels = greengarden.shape[1]

    return greenfactor, greengarden_percentage, greengarden_pixels


def create_filled_rasterarray(array, burn_polygon,
                              raster_path, rasterdata, print_rast):

    kwargs = {'projection': rasterdata['projection'],
              'geo_transform': rasterdata['geo_transform']}
    if 'no_data_value' in rasterdata:
        kwargs['no_data_value'] = rasterdata['no_data_value']

    with datasets.Dataset(array, **kwargs) as source:
        # set pixels outside geometry to 'no data'
        burn_value(dataset=source,
                   geometry=burn_polygon,
                   value=rasterdata['fillvalue'])
        # execute raster for debugging purposes
        if print_rast == 1:
            DRIVER_GDAL_GTIFF.CreateCopy(raster_path, source)

    return array


def prepare_aerial_array(tmp_aerial_tif,
                         aerial_image_path, projwin_garden, sr):

    # snap aerial image based on one building to geotiff
    ecw_gdal_translate(aerial_image_path, tmp_aerial_tif, projwin_garden)

    # inladen tmp aerial tiff for further calculation
    raster = gdal.Open(tmp_aerial_tif)
    rgbt = raster.ReadAsArray()
    geotransform = raster.GetGeoTransform()
    dtype = raster.GetRasterBand(1).DataType
    width = raster.RasterXSize
    height = raster.RasterYSize

    rasterdata = {'projection': osr.GetUserInputAsWKT(sr),
                  'geo_transform': geotransform,
                  # 'no_data_value': 1,
                  'data_type': dtype,
                  'sr': sr,
                  'width': width,
                  'height': height}

    return rgbt, rasterdata


def get_envelope_polygon(envelope_garden, sr):
    """ Return polygon for extent. """

    POLYGON = 'POLYGON (({x1} {y1},{x2} {y1},{x2} {y2},{x1} {y2},{x1} {y1}))'
    x1, y2, x2, y1 = envelope_garden
    wkt = POLYGON.format(x1=x1, y1=y1, x2=x2, y2=y2)

    return ogr.CreateGeometryFromWkt(wkt, sr)


def check_envelopes_input(garden_name, envelope_garden, envelope_aerial_image):

    g_x1, g_x2, g_y1, g_y2 = envelope_garden
    ai_x1, ai_x2, ai_y1, ai_y2 = envelope_aerial_image

    # check if polygon is within raster
    if not (g_x1 > ai_x1 and g_x2 < ai_x2 and g_y1 > ai_y1 and g_y2 < ai_y2):
        print(('[*] WARNING : Polygon (%s) is not (completely) '
               'within the aerial image envelope') % garden_name)

    # check size of garden
    area = (g_x2 - g_x1) * (g_y2 - g_y1)
    if area > 50000:
        skip_large_size = 1
        hint = '[*] WARNING : Skip polygon (%s) because size is too large: %s'
        print(hint % (garden_name, area))
    else:
        skip_large_size = 0

    return skip_large_size


def burn_value(dataset, geometry, value):
    """ Burn value where geometry is into dataset. """

    sr = geometry.GetSpatialReference()

    # put geometry into temporary layer
    datasource = DRIVER_OGR_MEMORY.CreateDataSource('')
    layer = datasource.CreateLayer('', sr)
    layer_defn = layer.GetLayerDefn()
    feature = ogr.Feature(layer_defn)
    feature.SetGeometry(geometry)
    layer.CreateFeature(feature)

    # burn no data
    burn_values = [value]
    gdal.RasterizeLayer(dataset, [1], layer, burn_values=burn_values)


def ecw_gdal_translate(aerial_image_path, tmp_aerial_tif, projwin_garden):
    """
    Fires a ``ecw_gdal_translate`` command on the
    :param aerial_image_path
    using ``subprocess``
    creating :param tmp_aerial_tif
    """
    a_srs = PROJ4_RD
    command = [
        # first comes the command
        'ecw-gdal_translate',
        # then come the arguments
        '-co', 'compress=deflate',
        '-ot', 'byte',
        '-projwin'
    ] + projwin_garden.split() + [
        '-a_srs', a_srs,
        aerial_image_path,
        tmp_aerial_tif,
    ]
    subprocess.check_output(command)


def gdalinfo(aerial_image_path):
    """
    Fires a ``ecw_gdalinfo`` command on the
    :param aerial_image_path
    using ``subprocess``
    """
    gdalinfolog = subprocess.check_output(['ecw-gdalinfo', aerial_image_path])
    grep_pixelsize_line = re.findall(
        "Pixel Size = \\([0-9+].[0-9]+", gdalinfolog,
    )[0]
    pixelsize = float(grep_pixelsize_line.split('(')[1])

    grep_upper_left_line = re.findall("Upper Left  \\((.*)\\)", gdalinfolog)[0]
    envelope_aerial_image = [None] * 4
    (envelope_aerial_image[0],
     envelope_aerial_image[3]) = map(float, grep_upper_left_line.split(','))

    grep_upper_left_line = re.findall("Lower Right \\((.*)\\)", gdalinfolog)[0]
    (envelope_aerial_image[1],
     envelope_aerial_image[2]) = map(float, grep_upper_left_line.split(','))

    return pixelsize, envelope_aerial_image


def get_projection(sr):
    """ Return simple userinput string for spatial reference, if any. """
    key = 'GEOGCS' if sr.IsGeographic() else 'PROJCS'
    name, code = sr.GetAuthorityName(key), sr.GetAuthorityCode(key)
    if name is None or code is None:
        return None
    return '{name}:{code}'.format(name=name, code=code)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    # add arguments here
    parser.add_argument(
        'aerial_image_path',
        metavar='AERIAL_IMAGE',
        help='input path with spaces to aerial image in ecw-format',
    )
    parser.add_argument(
        'gardens_path',
        metavar='POLYGON_PATH',
        help='path to polygon layer (e.g. gardens) with unique field',
    )
    parser.add_argument(
        'target_path',
        metavar='TARGET_PATH',
        help='output path gardens with green factor',
    )
    parser.add_argument(
        '--min_green',
        default=0.7,
        type=float,
        metavar='MINIMUM_GREEN',
        help='minimum greenfactor in aerial image (default: 0.7)',
    )
    parser.add_argument(
        '--max_green',
        default=0.9,
        type=float,
        metavar='MAXIMUM_GREEN',
        help='maximum greenfactor in aerial image (default: 0.9)',
    )
    parser.add_argument(
        '--pk',
        default='id',
        metavar='PRIMARY_KEY',
        help='select another primary key field (default: id)',
    )
    parser.add_argument(
        '--check_rast',
        action='store_true',
        help=('creates raster with the greenfactor to'
              ' check the min and max greenfactorsettings')
    )
    parser.add_argument(
        '-p',
        '--part',
        help='partial processing source, for example 2/3',
    )
    return parser


def main():
    """ Call command with args from parser. """
    kwargs = vars(get_parser().parse_args())
    return command(**kwargs)


if __name__ == '__main__':
    exit(main())
