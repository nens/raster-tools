# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
"""
Rasterize according to some index file from data in an ogr source to
raster files in AHN2 layout. Because of performance problems with the
ogr postgis driver, this module features its own datasource for postgis
connection strings, implemented using psycopg2.
"""

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

import psycopg2

from raster_tools import datasets

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

POLYGON = 'POLYGON (({x1} {y1},{x2} {y1},{x2} {y2},{x1} {y2},{x1} {y1}))'

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


def get_ogr_type(data_source, field_names):
    """
    Return the raster datatype corresponding to the field.
    """
    ogr_types = []
    for i, layer in enumerate(data_source):
        layer_defn = layer.GetLayerDefn()
        index = layer_defn.GetFieldIndex(field_names[i])
        field_defn = layer_defn.GetFieldDefn(index)
        ogr_types.append(field_defn.GetTypeName())
    if len(set(ogr_types)) > 1:
        print('Incompatible datatypes:')
        for i, layer in enumerate(data_source):
            print('{:<20} {:<10} {:<7}'.format(
                layer.GetName(), field_names[i], ogr_types[i],
            ))
        exit()
    return ogr_types[0]


def command(index_path, source_path, target_dir, attribute):
    """ Rasterize some postgis tables. """
    # investigate sources
    if source_path.lower().startswith('pg:'):
        source_data_source = PGDataSource(source_path)
    else:
        source_data_source = ogr.Open(source_path)
    if source_data_source.GetDriver().GetName() == 'ESRI Shapefile':
        # seems 1.9.1 does not sort, while 1.9.2 does
        ordered_source_data_source = sorted(source_data_source,
                                            key=lambda l: l.GetName())
    else:
        ordered_source_data_source = source_data_source

    source_field_names = []
    for source_layer in ordered_source_data_source:
        # check attribute for all source layers
        source_field_names.append(
            get_field_name(layer=source_layer, attribute=attribute)
        )
    ogr_type = get_ogr_type(
        data_source=ordered_source_data_source,
        field_names=source_field_names,
    )

    # Create indexes for shapefiles if necessary
    if source_data_source.GetDriver().GetName() == 'ESRI Shapefile':
        for source_layer in ordered_source_data_source:
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
            source_data_source.ExecuteSQL(
                b'CREATE SPATIAL INDEX ON {}'.format(source_layer_name),
            )

    # rasterize
    index_data_source = ogr.Open(index_path)
    index_layer = index_data_source[0]
    total = index_layer.GetFeatureCount()
    print('Starting rasterize.')
    gdal.TermProgress_nocb(0)
    for count, index_feature in enumerate(index_layer, 1):
        leaf_number = index_feature[b'BLADNR']
        target_path = os.path.join(
            target_dir, leaf_number[1:4], leaf_number + '.tif',
        )
        if os.path.exists(target_path):
            gdal.TermProgress_nocb(count / total)
            continue

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
        for i, source_layer in enumerate(ordered_source_data_source):
            source_field_name = source_field_names[i]
            source_layer.SetSpatialFilter(index_geometry)
            if not source_layer.GetFeatureCount():
                continue

            # create ogr layer if necessary
            if hasattr(source_layer, 'as_ogr_layer'):
                temp_data_source, source_layer = source_layer.as_ogr_layer(
                    name=source_field_name,
                    sr=index_layer.GetSpatialRef(),
                )

            # rasterize
            gdal.RasterizeLayer(
                dataset,
                [1],
                source_layer,
                options=['ATTRIBUTE={}'.format(source_field_name)]
            )
            burned = True

        if burned:
            leaf_number = index_feature[b'BLADNR']
            array = (dataset.ReadAsArray() == 255).astype('u1')
            if array.any():
                # save no data tif for inspection
                ndv_target_path = os.path.join(target_dir,
                                               'no_data',
                                               leaf_number[1:4],
                                               leaf_number + '.tif')
                try:
                    os.makedirs(os.path.dirname(ndv_target_path))
                except OSError:
                    pass
                array.shape = 1, dataset.RasterYSize, dataset.RasterXSize
                kwargs = {
                    'array': array,
                    'no_data_value': 0,
                    'geo_transform': get_geotransform(index_geometry),
                    'projection': osr.GetUserInputAsWKT(b'epsg:28992'),
                }
                with datasets.Dataset(**kwargs) as ndv_dataset:
                    DRIVER_GDAL_GTIFF.CreateCopy(ndv_target_path,
                                                 ndv_dataset,
                                                 options=['COMPRESS=DEFLATE'])

            # save
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
    parser = argparse.ArgumentParser(description=__doc__)
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


class PGFieldDefn(object):
    def __init__(self, name, typename):
        self.name = name
        self.typename = typename

    def GetName(self):
        return self.name

    def GetTypeName(self):
        return self.typename


class PGLayerDefn(object):
    def __init__(self, fields):
        self.names, self.typenames = zip(*fields)

    def GetFieldCount(self):
        return len(self.names)

    def GetFieldDefn(self, index):
        name = self.names[index]
        typename = self.typenames[index]
        return PGFieldDefn(name=name, typename=typename)

    def GetFieldIndex(self, name):
        return self.names.index(name)


class PGLayer(object):
    def __init__(self, connection, schema, table, geom_column='geom'):
        self.connection = connection
        self.schema = schema
        self.table = table
        self.geom_column = geom_column

    def GetName(self):
        return self.table

    def GetLayerDefn(self):
        typenames = {
            'integer': 'Integer',
            'real': 'Real',
        }
        sql = """
            select
                column_name,
                data_type
            from
                information_schema.columns
            where
                table_schema='{schema}'
                and table_name='{table}'
            order by
                ordinal_position
        """.format(schema=self.schema, table=self.table)
        cursor = self.connection.cursor()
        cursor.execute(sql)
        fields = [(r[0], typenames.get(r[1])) for r in cursor.fetchall()]
        return PGLayerDefn(fields=fields)

    def SetSpatialFilter(self, geometry):
        x1, x2, y1, y2 = geometry.GetEnvelope()
        self.box = POLYGON.format(x1=x1, y1=y1, x2=x2, y2=y2)

    def GetFeatureCount(self):
        sql = """
            select
                count(*)
            from
                {schema}.{table}
            where
                {geom_column} && ST_GeometryFromText('{box}')
        """.format(schema=self.schema, table=self.table,
                   box=self.box, geom_column=self.geom_column)
        cursor = self.connection.cursor()
        cursor.execute(sql)
        return cursor.fetchall()[0][0]

    def as_ogr_layer(self, name, sr):
        sql = """
            select
                ST_AsBinary(ST_Force2D({geom_column})),
                {name}
            from
                {schema}.{table}
            where
                {geom_column} && ST_GeometryFromText('{box}')
            """.format(name=name, schema=self.schema,
                       table=self.table, box=self.box,
                       geom_column=self.geom_column)

        cursor = self.connection.cursor()
        cursor.execute(sql)

        data_source = DRIVER_OGR_MEMORY.CreateDataSource('')
        layer = data_source.CreateLayer(b'', sr)
        layer.CreateField(ogr.FieldDefn(str(name), ogr.OFTInteger))
        layer_defn = layer.GetLayerDefn()
        for wkb, value in cursor:
            feature = ogr.Feature(layer_defn)
            feature[str(name)] = value
            try:
                feature.SetGeometry(ogr.CreateGeometryFromWkb(str(wkb)))
            except RuntimeError:
                pass
            layer.CreateFeature(feature)
        return data_source, layer


class PGDataSource(object):
    """
    Behaves like an ogr datasource, but actually uses psycopg2.
    """
    def __init__(self, source_path):
        info = dict(kv.split('=') for kv in source_path[3:].split())
        self.connection = psycopg2.connect(
            database=info['dbname'],
            host=info.get('host'),
            user=info.get('user'),
            password=info.get('password'),
        )
        self.schema = info.get('schemas')
        self.tables = info.get('tables').split(',')
        self.geom_column = info.get('geom_column', 'geom')

    """ Dummy driver """
    def GetDriver(self):
        class Driver(object):
            def GetName(self):
                return "PGDataSource"
        return Driver()

    def __iter__(self):
        for t in self.tables:
            l = PGLayer(
                connection=self.connection,
                schema=self.schema,
                table=t,
                geom_column=self.geom_column
            )
            yield l
