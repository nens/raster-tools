# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
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
        'real': ogr.OFTReal,
        'integer': ogr.OFTInteger,
        'numeric': ogr.OFTReal,
        'boolean': ogr.OFTBinary,
        'character varying': ogr.OFTString,
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
        SELECT DISTINCT ON ({geometry_column})
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

    def get_data(self, table, geometry, where=''):
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
        columns = ','.join(['ST_AsBinary(ST_Force2D({}))'.format(c)
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
        cursor.close()

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
                    geometry = ogr.CreateGeometryFromWkb(bytes(v))
                    feature.SetGeometry(geometry)
                else:
                    feature[k] = v
            layer.CreateFeature(feature)
        return data_source
