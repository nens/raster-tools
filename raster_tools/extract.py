# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import collections
import copy
import itertools
import json
import logging
import os
import Queue
import sys
import urllib
import urlparse
from multiprocessing import pool

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

Key = collections.namedtuple('Key', ['name', 'serial'])

"""
- Fix projection when -p epsg:3857
- Add operation model using ahn2:water and 15cm ahn2:bag
- Test larger models
"""


class Operation(object):
    """
    Base class for operations.
    """
    def __init__(self, **kwargs):
        """ An init that accepts kwargs. """
        self.kwargs = kwargs

    def _dataset(self, template):
        """
        Return dataset with dimensions, geo_transform and projection
        from template but data_type and no_data_value from self.
        """
        dataset = DRIVER_GDAL_MEM.Create(
            '',
            template.RasterXSize,
            template.RasterYSize,
            template.RasterCount,
            self.data_type,
        )
        dataset.SetProjection(template.GetProjection())
        dataset.SetGeoTransform(template.GetGeoTransform())
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(self.no_data_value)
        band.Fill(self.no_data_value)
        return dataset


class Elevation(Operation):
    """ Just store the elevation. """
    name = 'elevation'

    inputs = dict(elevation=dict(layers=['ahn2:int']))
    no_data_value = -9999
    data_type = 6

    def calculate(self, elevation):
        """ Return dataset. """
        result = self._dataset(elevation)
        # read
        data = elevation.ReadAsArray()
        # change no_data_values
        no_data_value = elevation.GetRasterBand(1).GetNoDataValue()
        mask = (data == no_data_value)
        data[mask] = self.no_data_value
        # write
        result.GetRasterBand(1).WriteArray(data)
        return result


class Preparation(object):
    """
    Preparation.
    """
    def __init__(self, path, layer, feature, **kwargs):
        """ Prepare a lot. """
        attribute = kwargs.pop('attribute')
        self.path = self._make_path(path, layer, feature, attribute)
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

        # Get resume value from dataset
        try:
            self.resume = int(self.dataset.GetMetadataItem(b'resume'))
            print('Resuming from block {}'.format(self.resume))
        except TypeError:
            self.resume = -1

        # put inputs properties on operation object
        for name, strategy in self.strategies.items():
            for key in ['no_data_value', 'data_type']:
                self.operation.inputs[name].update({key: strategy[key]})

        # debugging copy of indexes
        blocks_path = os.path.join(path, 'blocks.shp')
        if os.path.exists(blocks_path):
            DRIVER_OGR_SHAPE.DeleteDataSource(blocks_path)
        DRIVER_OGR_SHAPE.CopyDataSource(self.blocks, blocks_path)
        for name, chunks in self.chunks.items():
            chunks_path = os.path.join(path, name + '.shp')
            if os.path.exists(chunks_path):
                DRIVER_OGR_SHAPE.DeleteDataSource(chunks_path)
            DRIVER_OGR_SHAPE.CopyDataSource(chunks, chunks_path)

    def _make_path(self, path, layer, feature, attribute):
        """ Prepare a path from feature attribute or id. """
        try:
            model = feature[str(attribute)]
        except ValueError:
            model = layer.GetName() + str(feature.GetFID())

        print('Creating model: {}'.format(model))
        return os.path.join(path, model + '.tif')

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

    def _overlaps(self, polygon):
        """
        Return wether polygon is overlapping or contained by geometry.

        Ogr thinks contained polygons are not overlapping.
        """
        overlap = self.geometry.Overlaps(polygon)
        contain = self.geometry.Contains(polygon)
        return overlap or contain

    def _get_geoms(self, x1, y1, x2, y2):
        """ Return polygon, intersection tuple. """

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
                polygon = make_polygon(x1, y1, x2, y2)
                if not self._overlaps(polygon):
                    continue
                intersection = self.geometry.Intersection(polygon)

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
        x1, a, b, y2, c, d = self.dataset.GetGeoTransform()
        x2 = x1 + a + b
        y1 = y2 + c + d
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
                    polygon = make_polygon(x1, y1, x2, y2)
                    if not self._overlaps(polygon):
                        continue

                    # feature
                    feature = ogr.Feature(layer_defn)
                    feature[b'serial'] = serial.next()
                    feature[b'width'] = p2 - p1
                    feature[b'height'] = q2 - q1
                    feature.SetGeometry(polygon)
                    layer.CreateFeature(feature)
            return chunks

    def get_source(self):
        """ Return dictionary of source objects. """
        strategies = self.strategies
        inputs = self.operation.inputs
        blocks = DRIVER_OGR_MEMORY.CopyDataSource(self.blocks, '')
        blocks[0].SetAttributeFilter(b'serial>{}'.format(self.resume))
        source = Source(
            projection={k: strategies[k]['projection'] for k in strategies},
            layers={k: inputs[k]['layers'] for k in inputs},
            blocks=blocks,
            cellsize=self.cellsize,
            chunks=self.chunks,
            server=self.server,
        )
        return source

    def get_target(self):
        """ Return target object. """
        self.blocks[0].SetAttributeFilter(b'serial>{}'.format(self.resume))
        target = Target(chunks=self.chunks,
                        blocks=self.blocks,
                        dataset=self.dataset,
                        cellsize=self.cellsize,
                        operation=self.operation)
        return target


class Source(object):
    """
    Factory of source chunks.
    """
    def __init__(self, projection, cellsize, server, layers, chunks, blocks):
        """  """
        self.projection = projection
        self.cellsize = cellsize
        self.server = server
        self.layers = layers
        self.chunks = chunks
        self.blocks = blocks
        self.url = self._create_url()

    def _create_url(self):
        """ Build the general part of the url. """
        url = {}
        for name in self.projection:
            parameters = dict(
                layers=','.join(self.layers[name]),
                request='getgeotiff',
                compress='deflate',
                projection=self.projection[name],
            )
            url[name] = '{}?{}'.format(
                urlparse.urljoin(self.server, 'data'),
                urllib.urlencode(parameters),
            )
        return url

    def __getitem__(self, key):
        """ Get the chunk referenced by key. """
        # fetch chunk feature corresponding to key
        chunks = self.chunks[key.name][0]
        chunks.SetAttributeFilter(b'serial={}'.format(key.serial))
        feature = chunks.GetNextFeature()
        chunks.SetAttributeFilter(None)
        geometry = feature.geometry()

        # count blocks involved
        blocks = self.blocks[0]
        blocks.SetSpatialFilter(geometry.Buffer(-0.01 * min(self.cellsize)))
        refs = blocks.GetFeatureCount()

        # instantiate
        chunk = Chunk(key=key,
                      refs=refs,
                      url=self.url[key.name],
                      width=str(feature[b'width']),
                      height=str(feature[b'height']),
                      polygon=geometry.ExportToWkt())
        return chunk


class Chunk():
    """
    Represents a remote chunk of data.

    Ideally maps exactly to a remote storage chunk...
    """
    def __init__(self, key, refs, url, width, height, polygon):
        """ Prepare url. """
        parameters = dict(width=width, height=height, polygon=polygon)
        self.url = '{}&{}'.format(url, urllib.urlencode(parameters))
        self.refs = refs
        self.key = key

    def load(self):
        """ Load url into gdal dataset. """
        # retrieve file into gdal vsimem system
        vsi_path = '/vsimem/{}_{}'.format(*self.key)
        vsi_file = gdal.VSIFOpenL(str(vsi_path), b'w')
        url_file = urllib.urlopen(self.url)
        size = int(url_file.info().get('content-length'))
        gdal.VSIFWriteL(url_file.read(), size, 1, vsi_file)
        gdal.VSIFCloseL(vsi_file)

        # copy and remove
        dataset = gdal.Open(vsi_path)
        self.dataset = DRIVER_GDAL_MEM.CreateCopy('', dataset)
        dataset = None
        gdal.Unlink(vsi_path)


class Target(object):
    """
    Factory of target blocks
    """
    def __init__(self, blocks, chunks, dataset, cellsize, operation):
        self.blocks = blocks
        self.chunks = chunks
        self.dataset = dataset
        self.cellsize = cellsize
        self.operation = operation

    def __len__(self):
        """ Returns the featurecount. """
        return self.blocks[0].GetFeatureCount()

    def __iter__(self):
        """ Yields blocks. """
        blocks = self.blocks[0]
        blocks.ResetReading()
        for feature in blocks:

            # add the keys for the chunks
            geometry = feature.geometry()
            chunks = []
            for name in self.chunks:
                layer = self.chunks[name][0]
                layer.SetSpatialFilter(
                    geometry.Buffer(-0.01 * min(self.cellsize)),
                )
                chunks.extend([Key(name=name,
                                   serial=c[b'serial']) for c in layer])
                layer.SetSpatialFilter(None)

            # create the block object
            block = Block(chunks=chunks,
                          dataset=self.dataset,
                          attrs=feature.items(),
                          operation=self.operation,
                          geometry=feature.geometry().Clone())
            yield block


class Block(object):
    """ Self saving local chunk of data. """
    def __init__(self, attrs, dataset, geometry, operation, chunks):
        self.serial = attrs.pop('serial')
        self.chunks = chunks
        self.pixels = attrs
        self.dataset = dataset
        self.geometry = geometry
        self.operation = operation
        self.inputs = self._create_inputs()

    def __getitem__(self, key):
        """ Return corresponding value from inputs. """
        return self.inputs[key]

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

        return inputs

    def _polygon(self, dataset):
        """ Return polygon for dataset outline. """
        x1, a, b, y2, c, d = dataset.GetGeoTransform()
        w = dataset.RasterXSize
        h = dataset.RasterYSize
        x2 = x1 + w * a + h * b
        y1 = y2 + w * c + h * d
        return make_polygon(x1, y1, x2, y2)

    def _mask(self, dataset):
        """ Mask dataset where outside geometry. """
        wkt = dataset.GetProjection()
        polygon = self._polygon(dataset)
        difference = polygon.Difference(self.geometry)
        no_data_value = dataset.GetRasterBand(1).GetNoDataValue()
        datasource = DRIVER_OGR_MEMORY.CreateDataSource('')
        layer = datasource.CreateLayer(b'blocks', osr.SpatialReference(wkt))
        layer_defn = layer.GetLayerDefn()
        feature = ogr.Feature(layer_defn)
        feature.SetGeometry(difference)
        layer.CreateFeature(feature)
        gdal.RasterizeLayer(dataset, [1], layer, burn_values=[no_data_value])

    def _write(self, dataset):
        """ Write dataset into block. """
        p1 = self.pixels['p1']
        q1 = self.pixels['q1']
        self.dataset.WriteRaster(
            p1, q1, self.pixels['p2'] - p1, self.pixels['q2'] - q1,
            dataset.ReadRaster(0, 0, dataset.RasterXSize, dataset.RasterYSize),
        )

    def save(self):
        """
        Cut out and save block.
        """
        dataset = self.operation.calculate(**self.inputs)
        self._mask(dataset)
        self._write(dataset)
        self.dataset.SetMetadataItem(b'resume', str(self.serial))


def make_polygon(x1, y2, x2, y1):
    """ Return ogr wkb polygon for a rectangle. """
    polygon = ogr.CreateGeometryFromWkt(
        POLYGON.format(x1=x1, y1=y1, x2=x2, y2=y2),
    )
    return polygon


def load(toload, loaded):
    """ Load chunks until None comes out of the queue. """
    while True:
        # get
        chunk = toload.get()

        if chunk is None:
            break

        # load
        chunk.load()

        # put
        loaded.put(chunk)


def extract(preparation):
    """
    Extract for a single feature.
    """
    processes = 4
    toload = Queue.Queue(maxsize=processes)
    loaded = Queue.Queue()
    thread_pool = pool.ThreadPool(
        processes=processes,
        initializer=load,
        initargs=[toload, loaded],
    )

    loading = {}
    ready = {}
    blocks = {}

    target = preparation.get_target()
    source = preparation.get_source()
    iterator = iter(target)

    total = len(target)
    count = 0
    gdal.TermProgress_nocb(0)

    if total == 0:
        gdal.TermProgress_nocb(1)
        return

    while True:
        # see if any blocks left
        try:
            block = iterator.next()
        except StopIteration:
            block = None

        if block:
            # remember block
            blocks[block.serial] = block

            # add all related chunks
            for key in block.chunks:
                # skip if chunk already around
                if key in loading or key in ready:
                    continue
                chunk = source[key]
                toload.put(chunk)
                # remember chunk
                loading[key] = chunk

        #move chunks from loading to ready
        while True:
            try:
                chunk = loaded.get(block=False)
            except Queue.Empty:
                break
            key = chunk.key
            ready[key] = chunk
            del loading[key]

        # warp loaded chunks into blocks
        for block in blocks.values():
            for key in copy.copy(block.chunks):
                chunk = ready.get(key)
                if not chunk:
                    continue
                gdal.ReprojectImage(
                    chunk.dataset,
                    block[key.name],
                    None,
                    None,
                    gdal.GRA_NearestNeighbour,
                    0.0,
                    0.125,
                )
                #chunk administration and optional purge
                chunk.refs -= 1
                if not chunk.refs:
                    del ready[key]
                # block administration
                block.chunks.remove(key)

        # save blocks in serial order if possible
        while blocks:
            serial = min(blocks)
            block = blocks[serial]
            if block.chunks:
                break
            block.save()
            del blocks[serial]
            count += 1
            gdal.TermProgress_nocb(count / total)

        if not blocks:
            break

    # Signal the end for the processes
    for element in [None] * processes:
        toload.put(element)
    thread_pool.close()
    thread_pool.join()


def command(shape_path, target_dir, **kwargs):
    """
    Prepare and extract for each feature.
    """
    datasource = ogr.Open(shape_path)
    for layer in datasource:
        for feature in layer:
            preparation = Preparation(layer=layer,
                                      feature=feature,
                                      path=target_dir, **kwargs)
            extract(preparation)


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


if __name__ == '__main__':
    exit(main())
