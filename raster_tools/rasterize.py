# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import os
import sys

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

DRIVER_GDAL_GTIFF = gdal.GetDriverByName(b'gtiff')
DRIVER_GDAL_MEM = gdal.GetDriverByName(b'mem')
DRIVER_OGR_MEMORY = ogr.GetDriverByName(b'Memory')

NO_DATA_VALUE = {
    'Real': -3.4028234663852886e+38,
    'Integer': 255,
}

DATA_TYPE = {
    'Real': gdal.GDT_Float32,
    'Integer': gdal.GDT_Byte,
}

logger = logging.getLogger(__name__)
gdal.UseExceptions()
gdal.PushErrorHandler(b'CPLQuietErrorHandler')
ogr.UseExceptions()
osr.UseExceptions()


def get_geotransform(geometry, cellsize=(0.5, -0.5)):
    """ Return geotransform. """
    a, b, c, d = cellsize[0], 0.0, 0.0, cellsize[1]
    x1, x2, y1, y2 = geometry.GetEnvelope()
    return x1, a, b, y2, c, d


def get_field_name(layer, attribute):
    """
    Return the name of the sole field, or exit if there are more.
    """
    layer_name = layer.GetName()
    layer_defn = layer.GetLayerDefn()
    names = [layer_defn.GetFieldDefn(i).GetName().lower()
             for i in range(layer_defn.GetFieldCount())]
    choices = " or ".join([', '.join(names[:-1])] + names[-1:])
    # attribute given
    if attribute:
        if attribute.lower() in names:
            return attribute.lower()
        print('"{}" not in layer "{}". Choose from {}'.format(
            attribute, layer_name, choices
        ))
    # no attribute given
    else:
        if len(names) == 1:
            return names[0]
        elif not names:
            print('Layer "{}" has no attributes!')
        else:
            print('Layer "{}" has more than one attribute. Use -a option.\n'
                  'Available names: {}'.format(layer_name, choices))
    exit()


def get_ogr_type(datasource, field_names):
    """
    Return the raster datatype corresponding to the field.
    """
    ogr_types = []
    for i, layer in enumerate(datasource):
        layer_defn = layer.GetLayerDefn()
        index = layer_defn.GetFieldIndex(field_names[i])
        field_defn = layer_defn.GetFieldDefn(index)
        ogr_types.append(field_defn.GetTypeName())
    if len(set(ogr_types)) > 1:
        print('Incompatible datatypes:')
        for i, layer in enumerate(datasource):
            print('{:<20} {:<10} {:<7}'.format(
                layer.GetName(), field_names[i], ogr_types[i],
            ))
        exit()
    return ogr_types[0]


def command(index_path, source_path, target_dir, attribute):
    """ Do something spectacular. """
    # investigate sources
    #source_datasource = get_memory_copy(source_path)
    source_datasource = ogr.Open(source_path)
    if source_datasource.GetDriver().GetName() == 'ESRI Shapefile':
        # seems 1.9.1 does not sort, while 1.9.2 does
        ordered_source_datasource = sorted(source_datasource,
                                           key=lambda l: l.GetName())
    else:
        ordered_source_datasource = source_datasource

    source_field_names = []
    for source_layer in ordered_source_datasource:
        # check attribute for all source layers
        source_field_names.append(
            get_field_name(layer=source_layer, attribute=attribute)
        )
    ogr_type = get_ogr_type(
        datasource=ordered_source_datasource, field_names=source_field_names,
    )

    # Create indexes for shapefiles if necessary
    if source_datasource.GetDriver().GetName() == 'ESRI Shapefile':
        for source_layer in ordered_source_datasource:
            source_layer_name = source_layer.GetName()
            if os.path.isfile(source_path):
                source_layer_index_path = source_path[-4:] + '.qix'
            else:
                source_layer_index_path = os.path.join(
                    source_path, source_layer_name + '.qix',
                )
            if os.path.exists(source_layer_index_path):
                    continue
            print('Creating spatial index on {}.'.format(source_layer_name))
            source_datasource.ExecuteSQL(
                b'CREATE SPATIAL INDEX ON {}'.format(source_layer_name),
            )

    # rasterize
    index_datasource = ogr.Open(index_path)
    index_layer = index_datasource[0]
    total = index_layer.GetFeatureCount()
    print('Starting rasterize.')
    for count, index_feature in enumerate(index_layer, 1):
        index_geometry = index_feature.geometry()

        # prepare dataset
        data_type = DATA_TYPE[ogr_type]
        no_data_value = NO_DATA_VALUE[ogr_type]
        dataset = DRIVER_GDAL_MEM.Create('', 2000, 2500, 1, data_type)
        dataset.SetProjection(osr.GetUserInputAsWKT(b'epsg:28992'))
        dataset.SetGeoTransform(get_geotransform(index_geometry))
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(no_data_value)
        band.Fill(no_data_value)

        burned = False
        for i, source_layer in enumerate(ordered_source_datasource):
            source_field_name = source_field_names[i]
            # start experiment
            import psycopg2
            connection = psycopg2.connect(database='vector')
            cursor = connection.cursor()
            cursor.execute(
                ("select ST_AsBinary(geom), {} from data_verwerkt.top10_gras "
                 "where ST_GeometryFromText('{}') && geom").format(
                    source_field_name, index_geometry.ExportToWkt(),
                ),
            )
            temp_data_source = DRIVER_OGR_MEMORY.CreateDataSource('')
            temp_layer = temp_data_source.CreateLayer(
                b'', source_layer.GetSpatialRef(),
            )
            temp_layer.CreateField(ogr.FieldDefn(
                source_field_name, ogr.OFTInteger,
            ))
            temp_layer_defn = temp_layer.GetLayerDefn()
            for wkb, attr in cursor:
                temp_feature = ogr.Feature(temp_layer_defn)
                temp_feature[source_field_name] = attr
                temp_feature.SetGeometry(ogr.CreateGeometryFromWkb(str(wkb)))
                temp_layer.CreateFeature(temp_feature)
            source_layer = temp_layer
            # stop experiment
            source_layer.SetSpatialFilter(index_geometry)
            if not source_layer.GetFeatureCount():
                continue

            # rasterize
            gdal.RasterizeLayer(
                dataset,
                [1],
                source_layer,
                options=['ATTRIBUTE={}'.format(source_field_name)]
            )
            burned = True

        if burned:
            # save
            leaf_number = index_feature[b'BLADNR']
            target_path = os.path.join(
                target_dir, leaf_number[1:4], leaf_number + '.tif',
            )
            try:
                os.makedirs(os.path.dirname(target_path))
            except OSError:
                pass
            DRIVER_GDAL_GTIFF.CreateCopy(
                target_path, dataset, options=['COMPRESS=DEFLATE'],
            )

        gdal.TermProgress_nocb(count / total)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=(
        'Rasterize a vector source into multiple raster files.'
    ))
    parser.add_argument('index_path',
                        metavar='INDEX',
                        help='Path to ogr index')
    parser.add_argument('source_path',
                        metavar='SOURCE',
                        help='Path to ogr source')
    parser.add_argument('target_dir',
                        metavar='TARGET',
                        help='Output folder')
    parser.add_argument('-a', '--attribute',
                        help='Attribute to take burn value from.')
    return parser


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return command(**vars(get_parser().parse_args()))
