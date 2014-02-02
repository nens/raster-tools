# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import itertools
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

    inputs = dict(elevation=dict(layers=['elevation']))
    no_data_value = 3.4028235e+38
    data_type = 6

    def calculate(inputs):
        """ Return bytes. """
        return inputs['elevation'].ReadRaster()


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
        self.dataset = self._get_or_create_dataset()
        self.blocks = self._create_blocks()
        self.area = self._get_area()
        self.cell = self._get_cell()
        self.strategies = self._get_strategies()
        self.chunks = self._create_chunks()

        # put inputs properties on operation object
        for name, strategy in self.strategies.items():
            for key in ['no_data_value', 'data_type']:
                self.operation.inputs[name].update({key: strategy[key]})

        # debugging copy of indexes
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
        for name in self.operation.inputs:
            sources[name] = Source(
                projection=self.strategies[name]['projection'],
                layers=self.operation.inputs[name]['layers'],
                index=self.chunks[name][0],
                server=self.server,
                name=name,
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

    def _get_geoms(self, x1, y1, x2, y2):
        """ Return polygon, intersection tuple. """
        polygon = ogr.CreateGeometryFromWkt(
            POLYGON.format(x1=x1, y1=y1, x2=x2, y2=y2),
        )
        overlap = self.geometry.Overlaps(polygon)
        contain = self.geometry.Contains(polygon)
        if overlap or contain:
            intersection = self.geometry.Intersection(polygon)
        else:
            intersection = None
        return polygon, intersection

    def _create_blocks(self):
        """
        Create block index datasource.
        """
        # create datasource
        blocks = DRIVER_OGR_MEMORY.CreateDataSource('')
        wkt = osr.GetUserInputAsWKT(str(self.projection))
        layer = blocks.CreateLayer(b'blocks', osr.SpatialReference(wkt))
        layer.CreateField(ogr.FieldDefn(b'serial', ogr.OFTInteger))
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
        serial = itertools.count()
        for j in range(1 + (V - 1) // v):
            for i in range(1 + (U - 1) // u):
                # pixel indices and coordinates
                p1 = i * u
                q1 = j * v
                p2 = min(p1 + u, U)
                q2 = min(q1 + v, V)

                # geometries
                x1, y2 = p + a * p1 + b * q1, q + c * p1 + d * q1
                x2, y1 = p + a * p2 + b * q2, q + c * p2 + d * q2
                polygon, intersection = self._get_geoms(x1, y1, x2, y2)
                if intersection is None:
                    continue

                # feature
                feature = ogr.Feature(layer_defn)
                feature[b'serial'] = serial.next()
                feature[b'p1'] = p1
                feature[b'q1'] = q1
                feature[b'p2'] = p2
                feature[b'q2'] = q2
                feature.SetGeometry(intersection)
                layer.CreateFeature(feature)

        return blocks

    def _get_area(self):
        """ Return area envelope as wkt polygon. """
        x1, x2, y1, y2 = self.geometry.GetEnvelope()
        return POLYGON.format(x1=x1, y1=y1, x2=x2, y2=y2)

    def _get_cell(self):
        """ Return topleft cell as wkt polygon. """
        x1, dx, b, y2, c, dy = self.dataset.GetGeoTransform()
        x2 = x1 + dx
        y1 = y2 + dy
        return POLYGON.format(x1=x1, y1=y1, x2=x2, y2=y2)

    def _get_strategies(self):
        """ Return a dictionary with strategies. """
        strategies = {}
        for name in self.operation.inputs:
            parameters = dict(
                area=self.area,
                cell=self.cell,
                request='getstrategy',
                layers=','.join(self.operation.inputs[name]['layers']),
                projection=self.projection,
            )
            url = '{}?{}'.format(
                urlparse.urljoin(self.server, 'data'),
                urllib.urlencode(parameters)
            )
            strategies[name] = json.load(urllib.urlopen(url))
        return strategies

    def _adjust(self, width, height, before, after):
        """ Adjust width and height by geom ratios. """
        size = lambda x1, x2, y1, y2: (x2 - x1, y2 - y1)

        w1, h1 = size(*before.GetEnvelope())
        w2, h2 = size(*after.GetEnvelope())

        return int(round(width * w2 / w1)), int(round(height * h2 / h1))

    def _create_chunks(self):
        """ Return a dictionary with chunk index objects. """
        chunks = {}
        for name, strategy in self.strategies.items():

            # create datasource
            wkt = osr.GetUserInputAsWKT(str(strategy['projection']))
            chunks[name] = DRIVER_OGR_MEMORY.CreateDataSource('')
            layer = chunks[name].CreateLayer(
                b'chunks', osr.SpatialReference(wkt),
            )
            layer.CreateField(ogr.FieldDefn(b'serial', ogr.OFTInteger))
            layer.CreateField(ogr.FieldDefn(b'width', ogr.OFTInteger))
            layer.CreateField(ogr.FieldDefn(b'height', ogr.OFTInteger))
            layer_defn = layer.GetLayerDefn()

            # add features
            p, a, b, q, c, d = strategy['geo_transform']
            serial = itertools.count()
            for q1, q2 in strategy['chunks'][0]:
                for p1, p2 in strategy['chunks'][1]:
                    # geometries
                    x1, y2 = p + a * p1 + b * q1, q + c * p1 + d * q1
                    x2, y1 = p + a * p2 + b * q2, q + c * p2 + d * q2
                    polygon, intersection = self._get_geoms(x1, y1, x2, y2)
                    if intersection is None:
                        continue

                    # feature
                    width, height = self._adjust(
                        width=p2 - p1,
                        height=q2 - q1,
                        before=polygon,
                        after=intersection,
                    )
                    feature = ogr.Feature(layer_defn)
                    feature[b'serial'] = serial.next()
                    feature[b'width'] = width
                    feature[b'height'] = height
                    feature.SetGeometry(intersection)
                    layer.CreateFeature(feature)
            return chunks


class Source(object):
    """
    Factory of source chunks.
    """
    def __init__(self, projection, server, layers, index, name):
        """  """
        self.projection = projection
        self.server = server
        self.layers = layers
        self.index = index
        self.name = name
        self.url = self._create_url()

    def _create_url(self):
        """ Build the general part of the url. """
        parameters = dict(
            layers=','.join(self.layers),
            request='getgeotiff',
            compress='deflate',
            projection=self.projection,
        )
        url = '{}?{}'.format(
            urlparse.urljoin(self.server, 'data'),
            urllib.urlencode(parameters),
        )
        return url

    def chunks(self, block):
        """
        Returns chunk list for a geometry.
        """
        self.index.SetSpatialFilter(block.geometry)
        for feature in self.index:
            chunk = Chunk(source=self,
                          block=block,
                          attrs=feature.items(),
                          polygon=feature.geometry())
            yield chunk


class Chunk():
    """
    Represents a remote chunk of data.

    Ideally maps exactly to a remote storage chunk.
    """
    def __init__(self, source, block, attrs, polygon):
        """ Prepare url. """
        self.serial = attrs['serial']
        self.source = source
        self.block = block
        parameters = dict(
            width=str(attrs['width']),
            height=str(attrs['height']),
            polygon=polygon.ExportToWkt(),
        )
        self.url = '{}&{}'.format(
            self.source.url,
            urllib.urlencode(parameters)
        )

    def retrieve(self):
        """
        Load dataset from server or cache
        Caching happens at this level, if any.
        """
        cache.get
        url_file = urllib.urlopen(self.url)
        self.data = url_file.read()
    
    def convert(self):
        """ Copy bytes tif to gdal dataset. """
        vsi_file = gdal.VSIFOpenL('file', 'w')
        vsi_file.write(self.data)
        vsi_file.close()
        gdal.
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
            block = Block(dataset=self.dataset,
                          attrs=feature.items(),
                          operation=self.operation,
                          geometry=feature.geometry())
            yield block


class Block(object):
    """ Self saving local chunk of data. """
    def __init__(self, attrs, dataset, geometry, operation):
        self.serial = attrs.pop('serial')
        self.pixels = attrs
        self.dataset = dataset
        self.geometry = geometry
        self.operation = operation
        self.inputs = self._create_inputs()

    def _create_inputs(self):
        """ Create datasets for the operation. """
        inputs = {}
        for name in self.operation.inputs:
            # fancy sorting
            p1, p2, q1, q2 = (self.pixels[k] for k in sorted(self.pixels))

            # offset geo_transform
            p, a, b, q, c, d = self.dataset.GetGeoTransform()
            p = p + a * p1 + b * q1
            q = q + c * p1 + d * q1
            geo_transform = p, a, b, q, c, d

            # create dataset
            data_type = self.operation.inputs[name]['data_type']
            dataset = DRIVER_GDAL_MEM.Create(
                '', p2 - p1, q2 - q1, 1, data_type,
            )
            dataset.SetGeoTransform(geo_transform)
            dataset.SetProjection(self.dataset.GetProjection())
            no_data_value = self.operation.inputs[name]['no_data_value']
            band = dataset.GetRasterBand(1)
            band.SetNoDataValue(no_data_value)
            band.Fill(no_data_value)
            inputs[name] = dataset

    def save(self, string):
        """
        """
        p1 = self.pixels['p1']
        q1 = self.pixels['q1']
        band = self.dataset.GetRasterBand(1)
        band.WriteRaster(
            p1,
            q1,
            self.pixels['p2'] - p1,
            self.pixels['q2'] - q1,
            self.operation.calculate(self.layers),
        )


def extract(preparation):
    """
    Extract for a single feature.

    Plans:
        - Each chunk must know related blocks (and how many)
        - Each block must know related chunks (and how many)
        - Put n empty chunks on a task queue
        - Get any loaded chunks from a result queue
        - Get any related blocks
        - If now chunks loaded, do a loadstep yourself
        - Warp chunks into blocks, decrement counters
        - Check if any blocks are complete
        - Save those blocks, decrement counters
        - Check if any chunks are fully used
        - Remove from list
        - Have a threadpool do the loadsteps.
    Presto.
    """
    target = preparation.get_target()
    sources = preparation.get_sources()
    for block in target:
        for name in sources:
            for chunk in sources[name].chunks(block):
                chunk.load()
                gdal.ReprojectImage(chunk.dataset, block.inputs[name])
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
                        default='model',
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
