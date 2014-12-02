# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
"""
Extract layers from a raster server using a geometry.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import collections
import copy
import csv
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
from osgeo import gdalnumeric as np
from osgeo import ogr
from osgeo import osr

logger = logging.getLogger(__name__)
gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()
operations = {}

# Version management for outdated warning
VERSION = 12

GITHUB_URL = ('https://raw.github.com/nens/'
              'raster-tools/master/raster_tools/extract.py')

DRIVER_OGR_MEMORY = ogr.GetDriverByName(b'Memory')
DRIVER_OGR_SHAPE = ogr.GetDriverByName(b'ESRI Shapefile')
DRIVER_GDAL_MEM = gdal.GetDriverByName(b'mem')
DRIVER_GDAL_GTIFF = gdal.GetDriverByName(b'gtiff')

POLYGON = 'POLYGON (({x1} {y1},{x2} {y1},{x2} {y2},{x1} {y2},{x1} {y1}))'

Key = collections.namedtuple('Key', ['name', 'serial'])


class CompleteError(Exception):
    pass


class Operation(object):
    """
    Base class for operations.
    """


class Layers(Operation):
    """ Extract rasters according to a layer parameter. """
    name = 'layers'

    def __init__(self, layers, dtype, fillvalue, **kwargs):
        """ Initialize the operation. """
        self.layers = layers
        self.outputs = [self.name]
        self.inputs = {self.name: dict(layers=layers)}

        self.data_type = {
            self.name: dict(f4=gdal.GDT_Float32)[dtype],
        }

        if fillvalue is None:
            if dtype[0] == 'f':
                no_data_value = float(np.finfo(dtype).max)
            else:
                no_data_value = int(np.iinfo(dtype).max)
        else:
            no_data_value = np.array(fillvalue, dtype).tolist()
        self.no_data_value = {
            self.name: no_data_value,
        }

    def calculate(self, datasets):
        # create
        no_data_value = self.no_data_value[self.name]
        data_type = self.data_type[self.name]
        result = make_dataset(template=datasets[self.name],
                              data_type=data_type,
                              no_data_value=no_data_value)
        # read
        band = datasets[self.name].GetRasterBand(1)
        data = band.ReadAsArray().astype('f8')
        mask = ~band.GetMaskBand().ReadAsArray().astype('b1')
        data[mask] = no_data_value
        # write
        result.GetRasterBand(1).WriteArray(data)
        return {self.name: result}


class ThreeDi(Operation):
    """ Extract all rasters for a 3Di model. """
    name = '3di'

    # inputs
    I_SOIL = 'soil'
    I_LANDUSE = 'landuse'
    I_BATHYMETRY = 'bathymetry'

    # outputs
    O_SOIL = 'soil'
    O_CROP = 'crop'
    O_FRICTION = 'friction'
    O_BATHYMETRY = 'dem'
    O_INFILTRATION = 'infiltration'
    O_INTERCEPTION = 'interception'
    O_HYDRAULIC_CONDUCTIVITY = 'hydraulic_conductivity'

    no_data_value = {
        O_SOIL: -9999,
        O_CROP: -9999,
        O_FRICTION: -9999.,
        O_BATHYMETRY: -9999.,
        O_INFILTRATION: -9999.,
        O_INTERCEPTION: -9999.,
        O_HYDRAULIC_CONDUCTIVITY: -9999.,
    }

    data_type = {
        O_SOIL: gdal.GDT_Float32,
        O_CROP: gdal.GDT_Float32,
        O_FRICTION: gdal.GDT_Float32,
        O_BATHYMETRY: gdal.GDT_Float32,
        O_INFILTRATION: gdal.GDT_Float32,
        O_INTERCEPTION: gdal.GDT_Float32,
        O_HYDRAULIC_CONDUCTIVITY: gdal.GDT_Float32,
    }

    outputs = {
        O_SOIL: [I_SOIL],
        O_CROP: [I_LANDUSE],
        O_FRICTION: [I_LANDUSE],
        O_BATHYMETRY: [I_BATHYMETRY],
        O_INFILTRATION: [I_SOIL, I_LANDUSE],
        O_INTERCEPTION: [I_LANDUSE],
        O_HYDRAULIC_CONDUCTIVITY: [I_SOIL],
    }

    required = [y for x in outputs.values() for y in x]

    def __init__(self, floor, landuse, soil, **kwargs):
        """ Initialize the operation. """
        self.layers = {
            self.I_BATHYMETRY: dict(layers=','.join([
                'ahn2:int',
                'ahn2:bag!{}'.format(floor),
                'water:3di',
            ])),
            self.I_LANDUSE: dict(layers='use:wss-topleft'),
            self.I_SOIL: dict(layers='soil:3di'),
        }

        self.inputs = {k: v
                       for k, v in self.layers.items() if k in self.required}

        self.calculators = {
            self.O_SOIL: self._calculate_soil,
            self.O_CROP: self._calculate_crop,
            self.O_FRICTION: self._calculate_friction,
            self.O_BATHYMETRY: self._calculate_bathymetry,
            self.O_INFILTRATION: self._calculate_infiltration,
            self.O_INTERCEPTION: self._calculate_interception,
            self.O_HYDRAULIC_CONDUCTIVITY: self._calculate_hydr_cond,
        }

        self.soil_tables = self._get_soil_tables(soil)
        self.landuse_tables = self._get_landuse_tables(landuse)

    def _get_soil_tables(self, path):
        """
        Return conversion tables for landuse.
        """
        # check
        if not path:
            print('Operation {} requires a soil table.'.format(
                self.name,
            ))
            exit()
        # init
        intr_perm = [None] * 256
        max_infil = [None] * 256
        # fill
        with open(path) as soil_file:
            reader = csv.DictReader(soil_file, delimiter=b';')
            for record in reader:
                try:
                    code = int(record['Code'])
                except ValueError:
                    continue
                # intrinsic permeability
                field = 'Hydraulic_conductivity'
                try:
                    intr_perm[code] = float(record[field].replace(',', '.'))
                except ValueError:
                    print('Invalid {} for code {}: "{}"'.format(
                        field, code, record[field],
                    ))
                # max infiltration rate
                field = 'Max_infiltration_rate'
                try:
                    max_infil[code] = float(record[field].replace(',', '.'))
                except ValueError:
                    print('Invalid {} for code {}: "{}"'.format(
                        field, code, record[field],
                    ))
        return dict(max_infil=max_infil,
                    intr_perm=intr_perm)

    def _get_landuse_tables(self, path):
        """
        Return conversion tables for landuse.
        """
        # check
        if not path:
            print('Operation {} requires a landuse table.'.format(
                self.name,
            ))
            exit()
        # init
        friction = [None] * 256
        crop_type = [None] * 256
        interception = [None] * 256
        permeability = [None] * 256
        # fill
        with open(path) as landuse_file:
            reader = csv.DictReader(landuse_file, delimiter=b';')
            for record in reader:
                try:
                    code = int(record['Code'])
                except ValueError:
                    continue
                # friction
                field = 'Friction'
                try:
                    friction[code] = float(record[field].replace(',', '.'))
                except ValueError:
                    print('Invalid {} for code {}: "{}"'.format(
                        field, code, record[field],
                    ))
                # crop
                field = 'Crop_type'
                try:
                    crop_type[code] = int(record[field])
                except ValueError:
                    print('Invalid {} for code {}: "{}"'.format(
                        field, code, record[field],
                    ))
                # interception
                field = 'Interception'
                try:
                    interception[code] = float(record[field].replace(',', '.'))
                except ValueError:
                    print('Invalid {} for code {}: "{}"'.format(
                        field, code, record[field],
                    ))
                # permeability
                field = 'Permeability'
                try:
                    permeability[code] = float(record[field].replace(',', '.'))
                except ValueError:
                    print('Invalid {} for code {}: "{}"'.format(
                        field, code, record[field],
                    ))
        return dict(friction=friction,
                    crop_type=crop_type,
                    interception=interception,
                    permeability=permeability)

    def _calculate_soil(self, datasets):
        # short keys
        i = self.I_SOIL
        o = self.O_SOIL
        # create
        no_data_value = self.no_data_value[o]
        soil = make_dataset(template=datasets[i],
                            data_type=self.data_type[o],
                            no_data_value=no_data_value)
        # read
        band = datasets[i].GetRasterBand(1)
        data = band.ReadAsArray().astype('f8')
        mask = ~band.GetMaskBand().ReadAsArray().astype('b1')
        data[mask] = no_data_value
        # write
        soil.GetRasterBand(1).WriteArray(data)
        return soil

    def _calculate_crop(self, datasets):
        # short keys
        i = self.I_LANDUSE
        o = self.O_CROP
        # create
        no_data_value = self.no_data_value[o]
        crop = make_dataset(template=datasets[i],
                            data_type=self.data_type[o],
                            no_data_value=no_data_value)
        # read and convert
        conversion = np.array([no_data_value if x is None else x
                               for x in self.landuse_tables['crop_type']])
        band = datasets[i].GetRasterBand(1)
        data = conversion[band.ReadAsArray()]
        mask = ~band.GetMaskBand().ReadAsArray().astype('b1')
        data[mask] = no_data_value
        # write
        crop.GetRasterBand(1).WriteArray(data)
        return crop

    def _calculate_friction(self, datasets):
        # short keys
        i = self.I_LANDUSE
        o = self.O_FRICTION
        # create
        no_data_value = self.no_data_value[o]
        friction = make_dataset(template=datasets[i],
                                data_type=self.data_type[o],
                                no_data_value=no_data_value)
        # read and convert
        conversion = np.array([no_data_value if x is None else x
                               for x in self.landuse_tables['friction']])
        band = datasets[i].GetRasterBand(1)
        data = conversion[band.ReadAsArray()]
        mask = ~band.GetMaskBand().ReadAsArray().astype('b1')
        data[mask] = no_data_value
        # write
        friction.GetRasterBand(1).WriteArray(data)
        return friction

    def _calculate_bathymetry(self, datasets):
        # short keys
        i = self.I_BATHYMETRY
        o = self.O_BATHYMETRY
        # create
        no_data_value = self.no_data_value[o]
        bathymetry = make_dataset(template=datasets[i],
                                  data_type=self.data_type[o],
                                  no_data_value=no_data_value)
        # read
        band = datasets[i].GetRasterBand(1)
        data = band.ReadAsArray()
        mask = ~band.GetMaskBand().ReadAsArray().astype('b1')
        data[mask] = no_data_value
        # write
        bathymetry.GetRasterBand(1).WriteArray(data)
        return bathymetry

    def _calculate_infiltration(self, datasets):
        # short keys
        s = self.I_SOIL
        l = self.I_LANDUSE
        o = self.O_INFILTRATION
        # create
        no_data_value = self.no_data_value[o]
        infiltration = make_dataset(template=datasets[s],
                                    data_type=self.data_type[o],
                                    no_data_value=no_data_value)
        # read and convert soil
        s_conv = np.array([no_data_value if x is None else x
                           for x in self.soil_tables['max_infil']])
        s_band = datasets[s].GetRasterBand(1)
        s_data = s_conv[s_band.ReadAsArray()]
        s_mask = ~s_band.GetMaskBand().ReadAsArray().astype('b1')
        s_data[s_mask] = no_data_value
        # read and convert landuse
        l_conv = np.array([no_data_value if x is None else x
                           for x in self.landuse_tables['permeability']])
        l_band = datasets[l].GetRasterBand(1)
        l_data = l_conv[l_band.ReadAsArray()]
        l_mask = ~l_band.GetMaskBand().ReadAsArray().astype('b1')
        l_data[l_mask] = no_data_value
        # calculate
        data = np.where(
            np.logical_and(l_data != no_data_value, s_data != no_data_value),
            l_data * s_data,
            no_data_value,
        )
        # write
        infiltration.GetRasterBand(1).WriteArray(data)
        return infiltration

    def _calculate_interception(self, datasets):
        # short keys
        i = self.I_LANDUSE
        o = self.O_INTERCEPTION
        # create
        no_data_value = self.no_data_value[o]
        interception = make_dataset(template=datasets[i],
                                    data_type=self.data_type[o],
                                    no_data_value=no_data_value)
        # read and convert
        conversion = np.array([no_data_value if x is None else x
                               for x in self.landuse_tables['interception']])
        band = datasets[i].GetRasterBand(1)
        data = conversion[band.ReadAsArray()]
        mask = ~band.GetMaskBand().ReadAsArray().astype('b1')
        data[mask] = no_data_value
        # write
        interception.GetRasterBand(1).WriteArray(data)
        return interception

    def _calculate_hydr_cond(self, datasets):
        # short keys
        i = self.I_SOIL
        o = self.O_HYDRAULIC_CONDUCTIVITY
        # create
        no_data_value = self.no_data_value[o]
        permeability = make_dataset(template=datasets[i],
                                    data_type=self.data_type[o],
                                    no_data_value=no_data_value)
        # read and convert
        conversion = np.array([no_data_value if x is None else x
                               for x in self.soil_tables['intr_perm']])
        band = datasets[i].GetRasterBand(1)
        data = conversion[band.ReadAsArray()]
        mask = ~band.GetMaskBand().ReadAsArray().astype('b1')
        data[mask] = no_data_value
        # write
        permeability.GetRasterBand(1).WriteArray(data)
        return permeability

    def calculate(self, datasets):
        """ Return dictionary of output datasets. """
        # Temporarily change no data into 99 in landuse
        use_dataset = datasets[self.I_LANDUSE]
        use_band = use_dataset.GetRasterBand(1)
        use_array = use_band.ReadAsArray()
        use_no_data_value = use_band.GetNoDataValue()
        use_array[use_array == use_no_data_value] = 99
        use_band.WriteArray(use_array)

        # Temporarily change no data into 14 in soil
        soil_dataset = datasets[self.I_SOIL]
        soil_band = soil_dataset.GetRasterBand(1)
        soil_array = soil_band.ReadAsArray()
        soil_no_data_value = soil_band.GetNoDataValue()
        soil_array[soil_array == soil_no_data_value] = 14
        soil_band.WriteArray(soil_array)

        return {key: self.calculators[key](datasets) for key in self.outputs}


class Preparation(object):
    """
    Preparation.
    """
    def __init__(self, path, layer, feature, **kwargs):
        """ Prepare a lot. """
        attribute = kwargs.pop('attribute')
        self.server = kwargs.pop('server')
        self.operation = operations[kwargs.pop('operation')](**kwargs)

        self.information = self._get_information()
        self.projection = self._get_projection(kwargs.pop('projection'))
        self.cellsize = self._get_cellsize(kwargs.pop('cellsize'))

        self.wkt = osr.GetUserInputAsWKT(str(self.projection))
        self.sr = osr.SpatialReference(self.wkt)
        self.paths = self._make_paths(path, layer, feature, attribute)
        self.rpath = self.paths.pop('rpath')
        self.geometry = self._prepare_geometry(feature)
        self.datasets = self._get_or_create_datasets()

        # Get resume value
        try:
            with open(self.rpath) as resume_file:
                self.resume = int(resume_file.read())
        except IOError:
            self.resume = 0

        print(self.resume)

        self.index = self._create_index()

        # put inputs properties on operation object
        for name, strategy in self.strategies.items():
            for key in ['no_data_value', 'data_type']:
                self.operation.inputs[name].update({key: strategy[key]})

    def _make_paths(self, path, layer, feature, attribute):
        """ Prepare paths from feature attribute or id. """
        try:
            model = feature[str(attribute)]
        except ValueError:
            model = layer.GetName() + str(feature.GetFID())

        print('Creating model: {}'.format(model))
        root = os.path.join(path, model)
        paths = dict(rpath=os.path.join(root, 'resume.txt'))
        paths.update({n: os.path.join(root, '{}_{}.tif'.format(n, model))
                      for n in self.operation.outputs})
        return paths

    def _prepare_geometry(self, feature):
        """ Transform geometry if necessary. """
        geometry = feature.geometry()
        sr = geometry.GetSpatialReference()
        if sr:
            geometry.Transform(osr.CoordinateTransformation(sr, self.sr))
        return geometry

    def _get_information(self):
        """
        Return a dictionary with information about source stores.

        If a source consists of multiple stores, only the first store
        is considered.
        """
        information = {}
        for name, inp in self.operation.inputs.items():
            parameters = dict(
                request='getinfo',
                layer=inp['layers'].split(',')[0].split('!')[0],
            )
            url = '{}?{}'.format(
                urlparse.urljoin(self.server, 'data'),
                urllib.urlencode(parameters)
            )
            information[name] = json.load(urllib.urlopen(url))
        return information

    def _get_projection(self, projection):
        """
        Take appropriate projection from information or projection parameters.
        """
        if projection:
            return projection
        return self.information.values()[0]['projection']

    def _get_cellsize(self, cellsize):
        """
        Take appropriate cellsize from information or cellsize parameters.
        """
        if cellsize:
            return cellsize
        p, a, b, q, c, d = self.information.values()[0]['geo_transform']
        return a, -d

    def _create_dataset(self, name, path):
        """ """
        # dir
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass

        # properties
        a, b, c, d = self.cellsize[0], 0.0, 0.0, -self.cellsize[1]
        x1, x2, y1, y2 = self.geometry.GetEnvelope()
        p, q = a * (x1 // a), d * (y2 // d)

        width = -int((p - x2) // a)
        height = -int((q - y1) // d)
        geo_transform = p, a, b, q, c, d
        projection = self.wkt

        # create
        dataset = DRIVER_GDAL_GTIFF.Create(
            path, width, height, 1, self.operation.data_type[name],
            ['TILED=YES', 'BIGTIFF=YES', 'SPARSE_OK=TRUE', 'COMPRESS=DEFLATE'],
        )
        dataset.SetProjection(projection)
        dataset.SetGeoTransform(geo_transform)
        dataset.GetRasterBand(1).SetNoDataValue(
            self.operation.no_data_value[name],
        )
        return dataset

    def _get_or_create_datasets(self):
        """ Return a tif for each output. """
        datasets = {}
        for name, path in self.paths.items():
            if os.path.exists(path):
                datasets[name] = gdal.Open(path, gdal.GA_Update)
            else:
                datasets[name] = self._create_dataset(name, path)
        return datasets

    def _create_index(self):
        """ Create index object to take block geometries from. """
        self.index = Index(
            dataset=self.datasets.values()[0],
            geometry=self.geometry,
            resume=self.resume,
        )
        if self.index:
            if self.resume > 0:
                print('Resuming from block {}.'.format(self.resume))
        else:
            print('Already complete.')
            raise CompleteError()

    def _get_area(self):
        """ Return area envelope as wkt polygon. """
        x1, x2, y1, y2 = self.geometry.GetEnvelope()
        return POLYGON.format(x1=x1, y1=y1, x2=x2, y2=y2)

    def _get_cell(self):
        """ Return topleft cell as wkt polygon. """
        dataset = self.datasets.itervalues().next()
        x1, a, b, y2, c, d = dataset.GetGeoTransform()
        x2 = x1 + a + b
        y1 = y2 + c + d
        return POLYGON.format(x1=x1, y1=y1, x2=x2, y2=y2)

    def _create_chunks(self):
        """ Return a dictionary with chunk index objects. """
        chunks = {}
        for name, strategy in self.strategies.items():

            # create datasource
            wkt = osr.GetUserInputAsWKT(str(strategy['projection']))
            sr = osr.SpatialReference(wkt)
            chunks[name] = DRIVER_OGR_MEMORY.CreateDataSource('')
            layer = chunks[name].CreateLayer(b'chunks', sr)
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
                    polygon.AssignSpatialReference(sr)
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
        target = Target(rpath=self.rpath,
                        chunks=self.chunks,
                        blocks=self.blocks,
                        datasets=self.datasets,
                        cellsize=self.cellsize,
                        operation=self.operation)
        return target


class Index(object):
    """ Iterates the indices into the target dataset. """
    def __init__(self, dataset, geometry, resume):
        """
        Rasterize geometry into target dataset extent to find relevant
        blocks.
        """
        # make a dataset
        w, h = dataset.GetRasterBand(1).GetBlockSize()
        p, a, b, q, c, d = dataset.GetGeoTransform()
        index = DRIVER_GDAL_MEM.Create(
            '',
            (dataset.RasterXSize - 1) // w + 1,
            (dataset.RasterYSize - 1) // h + 1,
            1,
            gdal.GDT_Byte,
        )

        geo_transform = p, a * w, b * h, q, c * w, d * h
        index.SetProjection(dataset.GetProjection())
        index.SetGeoTransform(geo_transform)

        # rasterize where geometry is
        datasource = DRIVER_OGR_MEMORY.CreateDataSource('')
        sr = geometry.GetSpatialReference()
        layer = datasource.CreateLayer(b'geometry', sr)
        layer_defn = layer.GetLayerDefn()
        feature = ogr.Feature(layer_defn)
        feature.SetGeometry(geometry)
        layer.CreateFeature(feature)
        gdal.RasterizeLayer(
            index,
            [1],
            layer,
            burn_values=[1],
            options=['all_touched=true'],
        )

        # remember some of this
        self.sr = sr
        self.resume = resume
        self.block_size = w, h
        self.dataset_size = dataset.RasterXSize, dataset.RasterYSize
        self.geo_transform = dataset.GetGeoTransform()
        self.indices = index.ReadAsArray().nonzero()

    def get_indices(self, serial):
        """ Return indices into dataset. """
        w, h = self.block_size
        W, H = self.dataset_size
        y, x = self.indices[0][serial], self.indices[1][serial]
        x1 = w * x
        y1 = h * y
        x2 = min(W, (x + 1) * w)
        y2 = min(H, (y + 1) * h)
        return x1, y1, x2, y2

    def get_polygon(self, indices):
        """ Return ogr wkb polygon for a rectangle. """
        u1, v1, u2, v2 = indices
        p, a, b, q, c, d = self.geo_transform
        x1 = p + a * u1 + b * v1
        y1 = q + c * u1 + d * v1
        x2 = p + a * u2 + b * v2
        y2 = q + c * u2 + d * v2
        polygon = ogr.CreateGeometryFromWkt(
            POLYGON.format(x1=x1, y1=y1, x2=x2, y2=y2), self.sr,
        )
        return polygon

    def __len__(self):
        return len(self.indices[0])

    def __nonzero__(self):
        return len(self) > self.resume

    def __iter__(self):
        for serial in range(self.resume, len(self)):
            indices = self.get_indices(serial)
            yield {
                'serial': serial,
                'indices': indices,
                'polygon': self.get_polygon(indices),
            }


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
                layers=self.layers[name],
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
    def __init__(self, rpath, blocks, chunks, datasets, cellsize, operation):
        self.rpath = rpath
        self.blocks = blocks
        self.chunks = chunks
        self.datasets = datasets
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
            geometry = feature.geometry().Buffer(-0.01 * min(self.cellsize))
            geometry.AssignSpatialReference(self.blocks[0].GetSpatialRef())
            chunks = []
            for name in self.chunks:
                layer = self.chunks[name][0]
                clone = geometry.Clone()
                clone.Transform(osr.CoordinateTransformation(
                    clone.GetSpatialReference(),
                    layer.GetSpatialRef(),
                ))
                layer.SetSpatialFilter(clone)
                chunks.extend([Key(name=name,
                                   serial=c[b'serial']) for c in layer])
                layer.SetSpatialFilter(None)

            # create the block object
            block = Block(chunks=chunks,
                          rpath=self.rpath,
                          attrs=feature.items(),
                          datasets=self.datasets,
                          operation=self.operation,
                          geometry=feature.geometry().Clone())
            yield block


class Block(object):
    """ Self saving local chunk of data. """
    def __init__(self, attrs, datasets, geometry, operation, chunks, rpath):
        self.serial = attrs.pop('serial')
        self.chunks = chunks
        self.pixels = attrs
        self.rpath = rpath
        self.datasets = datasets
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
            bigdataset = self.datasets.itervalues().next()
            p, a, b, q, c, d = bigdataset.GetGeoTransform()
            p = p + a * p1 + b * q1
            q = q + c * p1 + d * q1
            geo_transform = p, a, b, q, c, d

            # create dataset
            data_type = self.operation.inputs[name]['data_type']
            dataset = DRIVER_GDAL_MEM.Create(
                '', p2 - p1, q2 - q1, 1, data_type,
            )
            dataset.SetGeoTransform(geo_transform)
            dataset.SetProjection(bigdataset.GetProjection())
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

    def _write(self, source, target):
        """ Write dataset into block. """
        p1 = self.pixels['p1']
        q1 = self.pixels['q1']
        target.WriteRaster(
            p1, q1, self.pixels['p2'] - p1, self.pixels['q2'] - q1,
            source.ReadRaster(0, 0, source.RasterXSize, source.RasterYSize),
        )

    def save(self):
        """
        Cut out and save block.
        """
        outputs = self.operation.calculate(self.inputs)
        for name in outputs:
            self._mask(outputs[name])
            self._write(source=outputs[name],
                        target=self.datasets[name])

        with open(self.rpath, 'w') as resume_file:
            resume_file.write(str(self.serial))


def make_dataset(template, data_type, no_data_value):
    """
    Return dataset with dimensions, geo_transform and projection
    from template but data_type and no_data_value from arguments.
    """
    dataset = DRIVER_GDAL_MEM.Create(
        '',
        template.RasterXSize,
        template.RasterYSize,
        template.RasterCount,
        data_type,
    )
    dataset.SetProjection(template.GetProjection())
    dataset.SetGeoTransform(template.GetGeoTransform())
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(no_data_value)
    band.Fill(no_data_value)
    return dataset


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
    processes = 8
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
            if not blocks:
                break

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

        # move chunks from loading to ready
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
                # chunk administration and optional purge
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

    # Signal the end for the processes
    for element in [None] * processes:
        toload.put(element)
    thread_pool.close()
    thread_pool.join()


def check_version():
    """
    Check if this is the highest available version of the script.
    """
    url_file = urllib.urlopen(GITHUB_URL)
    lines = url_file.readlines()
    url_file.close()

    for l in lines:
        if l.startswith('VERSION ='):
            remote_version = int(l.split('=')[-1].strip())
            break
    if remote_version > VERSION:
        print('This script is outdated. Get the latest at:\n{}'.format(
            GITHUB_URL,
        ))
        exit()


def command(shape_path, target_dir, **kwargs):
    """
    Prepare and extract for each feature.
    """
    if kwargs['version']:
        print('Extract script version: {}'.format(VERSION))
        exit()
    check_version()
    datasource = ogr.Open(shape_path)
    for layer in datasource:
        for feature in layer:
            try:
                preparation = Preparation(layer=layer,
                                          feature=feature,
                                          path=target_dir, **kwargs)
            except CompleteError:
                continue
            extract(preparation)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    # main
    parser.add_argument('shape_path',
                        metavar='SHAPE')
    parser.add_argument('target_dir',
                        metavar='OUTPUT')
    # options
    parser.add_argument('-v', '--version',
                        action='store_true')
    parser.add_argument('-s', '--server',
                        default='http://110-raster-d1.external-nens.local:5000')
    parser.add_argument('-o', '--operation',
                        default='3di',
                        choices=operations,
                        help='Type of output to be created. Default: 3di')
    parser.add_argument('-a', '--attribute',
                        default='model',
                        help='Attribute for tif filename. Default: model')
    parser.add_argument('-f', '--floor',
                        default=0.15,
                        type=float,
                        help='Floor height (3di). Default: 0.15')
    parser.add_argument('-c', '--cellsize',
                        type=float,
                        nargs=2,
                        help='Cellsize for output file.')
    parser.add_argument('-p', '--projection',
                        help='Output projection.')
    parser.add_argument('-tl', '--landuse',
                        help='Path to landuse csv.')
    parser.add_argument('-ts', '--soil',
                        help='Path to soil csv.')
    parser.add_argument('-l', '--layers',
                        help='Layers for layers operation.')
    parser.add_argument('-dt', '--dtype',
                        default='f4',
                        help='Datatype for layers operation')
    parser.add_argument('-fv', '--fillvalue',
                        help='Fillvalue for layers operation')
    return parser


def main():
    """ Call command with args from parser. """
    operations.update({cls.name: cls
                       for cls in Operation.__subclasses__()})
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return command(**vars(get_parser().parse_args()))


if __name__ == '__main__':
    exit(main())
