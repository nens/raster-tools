# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
"""
This module provides a routine to create a memory datasource
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

class PostgisSource(object):

    OGR_TYPES = {
        'real': ogr.OFTReal,
        'integer': ogr.OFTInteger,
        'numeric': ogr.OFTInteger,
        'character varying': ogr.OFTString,
    }

    SQL_DATA_TYPE = """
        SELECT
            column_name,
            data_type
        FROM
            information_schema.columns
        WHERE
            table_schema='{schema}'
            and table_name='{name}'
        ORDER BY
            ordinal_position
    """

    SQL_GEOMETRY_COLUMN = """
        SELECT
            f_geometry_column, srid
        FROM 
            geometry_columns
        WHERE 
            f_table_schema='{schema}' AND
            f_table_name='{name}'
    """

    SQL_DATA_SOURCE = """
        SELECT
            {columns}
        FROM
            {schema}.{name}
        WHERE
            {name}.{geometry_column} && ST_GeomFromWKB({wkb})
            {where}
    """

    def __init__(self, *args, **kwargs):
        """ """
        self.connection = psycopg2.connect(*args, **kwargs)

    def get_data(self, table, geometry, where):
        """ Return a section of a table as an ogr data source. """
        schema, name = table.split('.')
        cursor = self.connection.cursor()

        # get geometry column and srid
        sql = self.SQL_GEOMETRY_COLUMN.format(
            schema=schema,
            name=name,
        )
        cursor.execute(sql)
        geometry_column, srid = cursor.fetchall()[0]

        # get data types
        sql = self.SQL_DATA_TYPE.format(
            schema=schema,
            name=name,
        )
        cursor.execute(sql)
        all_columns, data_types = zip(*cursor.fetchall())

        # get records
        columns = ','.join(['ST_AsBinary({})'.format(c) 
                            if c == geometry_column 
                            else c
                            for c in all_columns])
        sql = self.SQL_DATA_SOURCE.format(
            name=name,
            where=where,
            schema=schema,
            columns=columns,
            geometry_column=geometry_column,
            wkb=psycopg2.Binary(geometry.ExportToWkb()),
        )
        cursor.execute(sql)
        records = cursor.fetchall()

        return dict(
            srid=srid,
            records=records,
            data_types=data_types,
            all_columns=all_columns,
            description=cursor.description,
            geometry_column=geometry_column,
        )

    def get_data_source(self, **kwargs):
        """ Return data as ogr data source. """
        data = self.get_data(**kwargs)
        geometry_column = data['geometry_column']
        
        # source and layer
        data_source = DRIVER_OGR_MEMORY.CreateDataSource('')
        spatial_ref = osr.SpatialReference()
        try:
            spatial_ref.ImportFromEPSG(data['srid'])
        except RuntimeError:
            spatial_ref.ImportFromEPSG(28992)
        
        # layer definition
        layer = data_source.CreateLayer(b'', spatial_ref)
        for n, t in zip(data['all_columns'], data['data_types']):
            if n == geometry_column:
                continue
            layer.CreateField(ogr.FieldDefn(n, self.OGR_TYPES[t]))
        layer_defn = layer.GetLayerDefn()

        # data insertion
        for r in data['records']:
            feature = ogr.Feature(layer_defn)
            for k, v in zip(data['all_columns'], r):
                if k == geometry_column:
                    import ipdb
                    ipdb.set_trace() 
                    try:
                        geometry = ogr.CreateGeometryFromWkb(str(v))
                    except RuntimeError:
                        continue
                    print('work')
                    feature.SetGeometry(geometry)
                else:
                    feature[k] = v
            layer.CreateFeature(feature)
        return data_source


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
    def __init__(self, connection, schema, table):
        self.connection = connection
        self.schema = schema
        self.table = table

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
                table_schema='{}'
                and table_name='{}'
            order by
                ordinal_position
        """.format(self.schema, self.table)
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
                {}.{}
            where
                geom && ST_GeometryFromText('{}')
        """.format(self.schema, self.table, self.box)
        cursor = self.connection.cursor()
        cursor.execute(sql)
        return cursor.fetchall()[0][0]

    def as_ogr_layer(self, name, sr):
        sql = """
            select
                ST_AsBinary(geom),
                {}
            from
                {}.{}
            where
                geom && ST_GeometryFromText('{}')
            """.format(name, self.schema, self.table, self.box)

        cursor = self.connection.cursor()
        cursor.execute(sql)

        data_source = DRIVER_OGR_MEMORY.CreateDataSource('')
        layer = data_source.CreateLayer(b'', sr)
        layer.CreateField(ogr.FieldDefn(name, ogr.OFTInteger))
        layer_defn = layer.GetLayerDefn()
        for wkb, value in cursor:
            feature = ogr.Feature(layer_defn)
            feature[name] = value
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
                table=t
            )
            yield l
