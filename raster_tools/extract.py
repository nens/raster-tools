# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import json
import logging
import os
import sys
import urllib
import urlparse

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

logger = logging.getLogger(__name__)
gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()
operations = {}

DRIVER_OGR_MEMORY = ogr.GetDriverByName(b'Memory')
DRIVER_OGR_SHAPE = ogr.GetDriverByName(b'ESRI Shapefile')
DRIVER_GDAL_MEM = gdal.GetDriverByName(b'mem')
DRIVER_GDAL_GTIFF = gdal.GetDriverByName(b'gtiff')

POLYGON = 'POLYGON (({x1} {y1},{x2} {y1},{x2} {y2},{x1} {y2},{x1} {y1}))'


class Operation(object):
    """
    Base class for operations.
    """
    def __init__(self, **kwargs):
        """ An init that accepts kwargs. """
        self.kwargs = kwargs


class Elevation(Operation):
    """ Just store the elevation. """
    name = 'elevation'

    layers = dict(elevation=['elevation'])
    no_data_value = 3.4028235e+38
    data_type = 6

    def calculate(datasets):
        """ Return bytes. """
        return datasets['elevation'].ReadRaster()


class Preparation(object):
    """
    Preparation.
    """
    def __init__(self, path, feature, **kwargs):
        """ Prepare a lot. """
        attribute = kwargs.pop('attribute')
        self.path = self._make_path(path, feature, attribute)
        self.server = kwargs.pop('server')
        self.cellsize = kwargs.pop('cellsize')
        self.projection = kwargs.pop('projection')
        self.operation = operations[kwargs.pop('operation')](**kwargs)
        self.geometry = self._prepare_geometry(feature)
        self.polygon = self._get_polygon()
        self.dataset = self._get_or_create_dataset()
        self.blocks = self._create_blocks()
        self.chunks = self._create_chunks()
        DRIVER_OGR_SHAPE.CopyDataSource(self.blocks,
                                        os.path.join(path, 'blocks.shp'))
        for name, chunks in self.chunks.items():
            DRIVER_OGR_SHAPE.CopyDataSource(chunks,
                                            os.path.join(path, name + '.shp'))

    def get_target(self):
        """ Return target object. """
        target = Target(
            index=self.blocks[0],
            dataset=self.dataset,
            operation=self.operation,
        )
        return target

    def get_sources(self):
        """ Return dictionary of source objects. """
        sources = {}
        for name, layers in self.operation.layers.items():
            sources[name] = Source(
                server=self.server,
                layers=layers,
                index=self.chunks[name][0],
            )
        return sources

    def _make_path(self, path, feature, attribute):
        """ Prepare a path from feature attribute or id. """
        if attribute:
            name = feature[str(attribute)] + '.tif'
        else:
            name = str(feature.GetFID()) + '.tif'
        return os.path.join(path, name)

    def _prepare_geometry(self, feature):
        """ Transform geometry if necessary. """
        geometry = feature.geometry()
        sr = geometry.GetSpatialReference()
        if sr:
            wkt = osr.GetUserInputAsWKT(str(self.projection))
            ct = osr.CoordinateTransformation(
                sr, osr.SpatialReference(wkt),
            )
            geometry.Transform(ct)
        return geometry

    def _get_polygon(self):
        """ Return envelope as polygon. """
        x1, x2, y1, y2 = self.geometry.GetEnvelope()
        return POLYGON.format(x1=x1, y1=y1, x2=x2, y2=y2)

    def _get_or_create_dataset(self):
        """ Create a tif and check against operation and index. """
        if os.path.exists(self.path):
            return gdal.Open(self.path, gdal.GA_Update)

        # dir
        try:
            os.makedirs(os.path.dirname(self.path))
        except OSError:
            pass

        # properties
        a, b, c, d = self.cellsize[0], 0.0, 0.0, -self.cellsize[1]
        x1, x2, y1, y2 = self.geometry.GetEnvelope()
        p, q = a * (x1 // a), d * (y2 // d)

        width = -int((p - x2) // a)
        height = -int((q - y1) // d)
        geo_transform = p, a, b, q, c, d
        projection = osr.GetUserInputAsWKT(str(self.projection))

        # create
        dataset = DRIVER_GDAL_GTIFF.Create(
            self.path, width, height, 1, self.operation.data_type,
            ['TILED=YES', 'BIGTIFF=YES', 'SPARSE_OK=TRUE', 'COMPRESS=DEFLATE'],
        )
        dataset.SetProjection(projection)
        dataset.SetGeoTransform(geo_transform)
        dataset.GetRasterBand(1).SetNoDataValue(self.operation.no_data_value)
        return dataset

    def _create_blocks(self):
        """
        Create block index datasource.
        """
        # create datasource
        blocks = DRIVER_OGR_MEMORY.CreateDataSource('')
        wkt = osr.GetUserInputAsWKT(str(self.projection))
        layer = blocks.CreateLayer(b'blocks', osr.SpatialReference(wkt))
        layer.CreateField(ogr.FieldDefn(b'p1', ogr.OFTInteger))
        layer.CreateField(ogr.FieldDefn(b'q1', ogr.OFTInteger))
        layer.CreateField(ogr.FieldDefn(b'p2', ogr.OFTInteger))
        layer.CreateField(ogr.FieldDefn(b'q2', ogr.OFTInteger))
        layer_defn = layer.GetLayerDefn()

        # add the polygons
        p, a, b, q, c, d = self.dataset.GetGeoTransform()
        u, v = self.dataset.GetRasterBand(1).GetBlockSize()
        U, V = self.dataset.RasterXSize, self.dataset.RasterYSize

        # add features
        for j in range(1 + (V - 1) // v):
            for i in range(1 + (U - 1) // u):
                # pixel indices and coordinates
                p1 = i * u
                q1 = j * v
                p2 = min(p1 + u, U)
                q2 = min(q1 + v, V)

                # polygon
                x1, y2 = p + a * p1 + b * q1, q + c * p1 + d * q1
                x2, y1 = p + a * p2 + b * q2, q + c * p2 + d * q2
                polygon = ogr.CreateGeometryFromWkt(
                    POLYGON.format(x1=x1, y1=y1, x2=x2, y2=y2),
                )
                intersection = self.geometry.Intersection(polygon)
                if not intersection.GetGeometryCount():
                    continue

                # feature
                feature = ogr.Feature(layer_defn)
                feature[b'p1'] = p1
                feature[b'q1'] = q1
                feature[b'p2'] = p2
                feature[b'q2'] = q2
                feature.SetGeometry(intersection)
                layer.CreateFeature(feature)

        return blocks

    def _create_chunks(self):
        """ Return a dictionary with chunk index objects. """
        result = {}
        for name, layers in self.operation.layers.items():
            parameters = dict(
                layers=','.join(layers),
                request='getstrategy',
                polygon=self.polygon,
                projection=self.projection,
            )
            url = '{}?{}'.format(
                urlparse.urljoin(self.server, 'data'),
                urllib.urlencode(parameters)
            )
            strategy = json.load(urllib.urlopen(url))

            # create datasource
            chunks = DRIVER_OGR_MEMORY.CreateDataSource('')
            wkt = osr.GetUserInputAsWKT(str(strategy['projection']))
            layer = chunks.CreateLayer(b'chunks', osr.SpatialReference(wkt))
            layer_defn = layer.GetLayerDefn()

            # add the polygons
            p, a, b, q, c, d = strategy['geo_transform']
            u, v = strategy['chunks'][1:]

            # add features
            for q1, q2 in strategy['blocks'][0]:
                for p1, p2 in strategy['blocks'][1]:
                    # polygon
                    x1, y2 = p + a * p1 + b * q1, q + c * p1 + d * q1
                    x2, y1 = p + a * p2 + b * q2, q + c * p2 + d * q2
                    polygon = ogr.CreateGeometryFromWkt(
                        POLYGON.format(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                    # intersection
                    intersection = self.geometry.Intersection(polygon)
                    if not intersection.GetGeometryCount():
                        continue
                    # feature
                    feature = ogr.Feature(layer_defn)
                    feature.SetGeometry(intersection)
                    layer.CreateFeature(feature)
            result[name] = chunks
            return result


class Source(object):
    """
    Factory of source chunks.
    """
    def __init__(self, index, server, layers):
        """  """
        self.server = server
        self.index = index
        self.layers = layers

    def get_chunks(self, geometry):
        """
        Returns chunk list for a geometry.
        """
        # transform geometry to own sr
        # set filter
        # return chunks
        pass


class Chunk():
    """
    Represents a remote chunk of data.

    Ideally maps exactly to a remote storage chunk.
    """
    def __init__(self, width, height, layers, server, polygon, projection):
        """ Prepare url. """
        parameters = dict(
            width=str(width),
            height=str(width),
            layers=','.join(layers),
            request='getgeotiff',
            compress='deflate',
            polygon=polygon.ExportToWkt(),
            projection=projection,
        )
        self.url = '{}?{}'.format(
            server,
            urllib.urlencode(parameters)
        )

    def load(self):
        """
        Load dataset from server.
        Caching happens at this level, if any.
        """
        url_file = urllib.urlopen(self.url)
        vsi_file = gdal.VSIFOpenL('myfile', 'w')
        vsi_file.write(url_file.read())
        vsi_file.close()
        # now what?
        self.dataset = None


class Target(object):
    """
    Factory of target blocks
    """
    def __init__(self, index, dataset, operation):
        self.index = index
        self.dataset = dataset
        self.operation = operation

    def __iter__(self):
        """ Yields blocks. """
        self.index.ResetReading()
        for feature in self.index:
            block = Block(
                attrs=feature.items(),
                dataset=self.dataset,
                geometry=feature.geometry(),
                operation=self.operation,
            )
            yield block


class Block(object):
    """ Self saving local chunk of data. """
    def __init__(self, attrs, dataset, geometry, operation):
        self.attrs = attrs
        self.dataset = dataset
        self.geometry = geometry
        self.operation = operation
        self.layers = self._create_layers()

    def _create_layers:
        """ Create datasets for the operation. """
        layers = []
        for l in self.operation.layers:
            p, a, b, q, c, d = self.dataset.GetGeoTransform()
            p, q = p + a * attrs['p0'] 
            DRIVER_GDAL_MEM.Create(
                


    def save(string):
        """
        """
        p1 = self.attrs['p1']
        q1 = self.attrs['q1']
        band = self.GetRasterBand(1)
        band.WriteRaster(
            p1,
            q1,
            self.attrs['p2'] - p1,
            self.attrs['q2'] - q1,
            self.operation.calculate(self.layers),
        )




def extract(preparation):
    """
    Extract for a single feature.
    """
    target = preparation.get_target()
    sources = preparation.get_sources()
    for block in target:
        print(block)
        exit()
        for name, source in sources.items():
            for chunk in source.chunks(block.geometry):
                chunk.load()
                gdal.ReprojectImage(chunk.dataset, block[name])
        block.save()


def command(shape_path, target_dir, **kwargs):
    """
    Prepare and extract for each feature.
    """
    datasource = ogr.Open(shape_path)
    layer = datasource[0]
    for feature in layer:
        preparation = Preparation(feature=feature, path=target_dir, **kwargs)
        extract(preparation)
        break


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # main
    parser.add_argument('shape_path',
                        metavar='SHAPE')
    parser.add_argument('target_dir',
                        metavar='GTIFF')
    # options
    parser.add_argument('-s', '--server',
                        default='https://raster.lizard.net')
    parser.add_argument('-o', '--operation',
                        default='elevation',
                        help='Operation')
    parser.add_argument('-a', '--attribute',
                        default='Model',
                        help='Attribute for tif filename.')
    parser.add_argument('-f', '--floor',
                        default=0.15,
                        help='Floor elevation above ground level')
    parser.add_argument('-c', '--cellsize',
                        default=[0.5, 0.5],
                        type=float,
                        nargs=2,
                        help='Cellsize for output file')
    parser.add_argument('-p', '--projection',
                        default='epsg:28992',
                        help='Spatial reference system for output file.')
    return parser


def main():
    """ Call command with args from parser. """
    operations.update({cls.name: cls
                       for cls in Operation.__subclasses__()})
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return command(**vars(get_parser().parse_args()))
