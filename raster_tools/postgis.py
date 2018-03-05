# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.
"""
This module provides a routine to create a memory datasource
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from osgeo import ogr
from osgeo import osr

import psycopg2

DRIVER_OGR_MEMORY = ogr.GetDriverByName(b'Memory')

ogr.UseExceptions()
osr.UseExceptions()


class PostgisSource(object):

    OGR_TYPES = {
        # boolean
        'boolean': ogr.OFTBinary,
        # integer
        'bigint': ogr.OFTInteger,
        'integer': ogr.OFTInteger,
        # real
        'real': ogr.OFTReal,
        'numeric': ogr.OFTReal,
        'double precision': ogr.OFTReal,
        # string
        'date': ogr.OFTString,
        'text': ogr.OFTString,
        'USER-DEFINED': ogr.OFTString,
        'character varying': ogr.OFTString,
        'timestamp with time zone': ogr.OFTString,
        'timestamp without time zone': ogr.OFTString,
    }

    SQL_GEOMETRY_COLUMN = """
        SELECT
            f_geometry_column, srid
        FROM
            geometry_columns
        WHERE
            f_table_schema='{schema}' AND
            f_table_name='{name}'
    """

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

    SQL_DATA_SOURCE = """
        SELECT
            {columns}
        FROM
            {schema}.{name}
        WHERE
            {geom} && {request} AND ST_Intersects({geom}, {request})
    """

    def __init__(self, *args, **kwargs):
        """ """
        self.connection = psycopg2.connect(*args, **kwargs)

    def _get_srid(self, geometry):
        sr = geometry.GetSpatialReference()
        key = str('GEOGCS') if sr.IsGeographic() else str('PROJCS')
        srid = sr.GetAuthorityCode(key)
        if srid is None:
            print('Geometry spatial reference lacks authority code.')
            exit()
        return srid

    def _get_data(self, table, geometry):
        """ Return a section of a table as an ogr data source. """
        try:
            schema, name = table.split('.')
        except ValueError:
            schema, name = 'public', table

        cursor = self.connection.cursor()

        # get geometry column and srid
        sql = self.SQL_GEOMETRY_COLUMN.format(
            schema=schema,
            name=name,
        )
        cursor.execute(sql)
        geom, srid = cursor.fetchall()[0]

        # get data types
        sql = self.SQL_DATA_TYPE.format(
            schema=schema,
            name=name,
        )
        cursor.execute(sql)
        column_names, data_types = zip(*cursor.fetchall())
        columns = str(',').join(column_names)

        # request
        template = 'ST_Transform(ST_SetSRID(ST_GeomFromWKB({}), {}), {})'
        wkb = psycopg2.Binary(geometry.ExportToWkb())
        _srid = self._get_srid(geometry)
        request = template.format(wkb, _srid, srid)

        replace = 'ST_AsBinary(ST_Transform({}, {}))'.format(geom, _srid)
        columns = ','.join(column_names).replace(geom, replace)

        # get data
        sql = self.SQL_DATA_SOURCE.format(
            geom=geom,
            name=name,
            schema=schema,
            columns=columns,
            request=request,
        )
        cursor.execute(sql)
        records = cursor.fetchall()
        cursor.close()

        return dict(
            geom=geom,
            srid=srid,
            records=records,
            data_types=data_types,
            column_names=column_names,
            description=cursor.description,
        )

    def get_data_source(self, name='', **kwargs):
        """ Return data as ogr data source. """
        data = self._get_data(**kwargs)
        geom = data['geom']

        # source and layer
        data_source = DRIVER_OGR_MEMORY.CreateDataSource('')
        spatial_ref = kwargs['geometry'].GetSpatialReference()

        # layer definition
        layer = data_source.CreateLayer(str(name), spatial_ref)
        for n, t in zip(data['column_names'], data['data_types']):
            if n == geom:
                continue
            layer.CreateField(ogr.FieldDefn(n, self.OGR_TYPES[t]))
        layer_defn = layer.GetLayerDefn()

        # data insertion
        g = data['column_names'].index(geom)
        for r in data['records']:
            # geometry
            try:
                geometry = ogr.CreateGeometryFromWkb(bytes(r[g]))
            except RuntimeError:
                # there were geometries giving trouble on trusty gdal
                print('Skipping geometry:')
                print(bytes(r[g]))
                continue
            feature = ogr.Feature(layer_defn)
            feature.SetGeometry(geometry)

            # attributes
            for i, (n, v) in enumerate(zip(data['column_names'], r)):
                if i == g:
                    continue
                feature[n] = v
            layer.CreateFeature(feature)
        return data_source
