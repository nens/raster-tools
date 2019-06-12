# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.
"""
Extract layers from a raster server using a geometry.

Main operation is to build and save a rudimentary 2D part for a 3Di model.
Alternatively it is possible to extract from a single layer on the remote
server.
"""

import argparse
import collections
import csv
import os
import threading

from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.parse import urljoin

import queue as queues

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from raster_tools import datasources
from raster_tools import datasets

import numpy as np

operations = {}

# Version management for outdated warning
VERSION = 30

GITHUB_URL = ('https://raw.github.com/nens/'
              'raster-tools/master/raster_tools/extract.py')

DRIVER_OGR_MEMORY = ogr.GetDriverByName('Memory')
DRIVER_OGR_SHAPE = ogr.GetDriverByName('ESRI Shapefile')
DRIVER_GDAL_MEM = gdal.GetDriverByName('mem')
DRIVER_GDAL_GTIFF = gdal.GetDriverByName('gtiff')
DTYPES = {'u1': gdal.GDT_Byte,
          'u2': gdal.GDT_UInt16,
          'u4': gdal.GDT_UInt32,
          'i2': gdal.GDT_Int16,
          'i4': gdal.GDT_Int32,
          'f4': gdal.GDT_Float32}

POLYGON = 'POLYGON (({x1} {y1},{x2} {y1},{x2} {y2},{x1} {y2},{x1} {y1}))'

# argument defaults
ATTRIBUTE = 'model'
CELLSIZE = 0.5, 0.5
DTYPE = 'f4'
FLOOR = 0.15
OPERATION = '3di-ahn2'
PROJECTION = 'EPSG:28992'
TIMESTAMP = '1970-01-01T00:00:00Z'
SERVER = 'https://raster.lizard.net'

Tile = collections.namedtuple('Tile', ['width',
                                       'height',
                                       'origin',
                                       'serial',
                                       'polygon'])


class CompleteError(Exception):
    pass


class Operation:
    """
    Base class for operations.
    """


class Layers(Operation):
    """ Extract rasters according to a layer parameter. """
    name = 'layers'

    def __init__(self, layers, dtype, fillvalue, time, **kwargs):
        """ Initialize the operation. """
        # self.layers = layers
        self.outputs = {self.name: [self.name]}
        self.inputs = {self.name: {'layers': layers, 'time': time}}
        self.data_type = {self.name: DTYPES[dtype]}
        self.dtype = dtype

        if fillvalue is None:
            if dtype[0] == 'f':
                no_data_value = float(np.finfo(dtype).max)
            else:
                no_data_value = int(np.iinfo(dtype).max)
        else:
            no_data_value = np.array(fillvalue, dtype).tolist()
        self.no_data_value = {self.name: no_data_value}

    def calculate(self, datasets):
        dataset = datasets[self.name]
        band = dataset.GetRasterBand(1)
        return {self.name: {
            'active': band.GetMaskBand().ReadAsArray(),
            'values': band.ReadAsArray().astype(self.dtype),
            'no_data_value': self.no_data_value[self.name],
            'geo_transform': dataset.GetGeoTransform(),
            'projection': dataset.GetProjection(),
        }}


class ThreeDiBase:
    """ Extract all rasters for a 3Di model. """
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

    required = set([y for x in outputs.values() for y in x])

    def __init__(self, floor, landuse, soil, time, **kwargs):
        """
        Initialize the operation. The subclasses have to set the layers
        attribute.
        """
        # create the inputs that will be loaded from the server
        self.inputs = {}
        for k in self.layers:
            if k in self.required:
                self.inputs[k] = {'time': time}
                self.inputs[k].update(self.layers[k])

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
            reader = csv.DictReader(soil_file, delimiter=';')
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
            reader = csv.DictReader(landuse_file, delimiter=';')
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

        dataset = datasets[i]
        band = dataset.GetRasterBand(1)

        return {
            'values': band.ReadAsArray().astype(DTYPE),
            'active': band.GetMaskBand().ReadAsArray(),
            'geo_transform': dataset.GetGeoTransform(),
            'projection': dataset.GetProjection(),
            'no_data_value': self.no_data_value[o],
        }

    def _calculate_crop(self, datasets):
        # short keys
        i = self.I_LANDUSE
        o = self.O_CROP

        # read and convert
        no_data_value = self.no_data_value[o]
        dataset = datasets[i]
        band = dataset.GetRasterBand(1)
        conversion = np.array([no_data_value if x is None else x
                               for x in self.landuse_tables['crop_type']])

        return {
            'values': conversion[band.ReadAsArray()].astype(DTYPE),
            'active': band.GetMaskBand().ReadAsArray(),
            'geo_transform': dataset.GetGeoTransform(),
            'projection': dataset.GetProjection(),
            'no_data_value': no_data_value,
        }

    def _calculate_friction(self, datasets):
        # short keys
        i = self.I_LANDUSE
        o = self.O_FRICTION

        # read and convert
        no_data_value = self.no_data_value[o]
        dataset = datasets[i]
        conversion = np.array([no_data_value if x is None else x
                               for x in self.landuse_tables['friction']])
        band = dataset.GetRasterBand(1)

        return {
            'values': conversion[band.ReadAsArray()].astype(DTYPE),
            'active': band.GetMaskBand().ReadAsArray(),
            'geo_transform': dataset.GetGeoTransform(),
            'projection': dataset.GetProjection(),
            'no_data_value': no_data_value,
        }

    def _calculate_bathymetry(self, datasets):
        # short keys
        i = self.I_BATHYMETRY
        o = self.O_BATHYMETRY

        # read
        dataset = datasets[i]
        band = dataset.GetRasterBand(1)
        values = band.ReadAsArray().astype(DTYPE)
        active = band.GetMaskBand().ReadAsArray()

        # mask nan (for example when floor is nan)
        active[np.isnan(values)] = False

        return {
            'values': values,
            'active': active,
            'geo_transform': dataset.GetGeoTransform(),
            'projection': dataset.GetProjection(),
            'no_data_value': self.no_data_value[o],
        }

    def _calculate_infiltration(self, datasets):
        # short keys
        s = self.I_SOIL
        c = self.I_LANDUSE
        o = self.O_INFILTRATION

        # read and convert soil
        no_data_value = self.no_data_value[o]
        s_conv = np.array([no_data_value if x is None else x
                           for x in self.soil_tables['max_infil']])
        s_dataset = datasets[s]
        s_band = s_dataset.GetRasterBand(1)
        s_data = s_conv[s_band.ReadAsArray()]
        s_mask = ~s_band.GetMaskBand().ReadAsArray(),
        s_data[s_mask] = no_data_value

        # read and convert landuse
        c_conv = np.array([no_data_value if x is None else x
                           for x in self.landuse_tables['permeability']])
        c_dataset = datasets[c]
        c_band = c_dataset.GetRasterBand(1)
        c_data = c_conv[c_band.ReadAsArray()]
        c_mask = ~c_band.GetMaskBand().ReadAsArray(),
        c_data[c_mask] = no_data_value

        # calculate
        values = np.where(
            np.logical_and(c_data != no_data_value, s_data != no_data_value),
            c_data * s_data,
            no_data_value,
        ).astype(DTYPE)

        return {
            'values': values,
            'active': np.where(values == no_data_value, 0, 255).astype('u1'),
            'geo_transform': s_dataset.GetGeoTransform(),
            'projection': s_dataset.GetProjection(),
            'no_data_value': no_data_value,
        }

    def _calculate_interception(self, datasets):
        # short keys
        i = self.I_LANDUSE
        o = self.O_INTERCEPTION

        # read and convert
        no_data_value = self.no_data_value[o]
        conversion = np.array([no_data_value if x is None else x
                               for x in self.landuse_tables['interception']])
        dataset = datasets[i]
        band = dataset.GetRasterBand(1)

        return {
            'values': conversion[band.ReadAsArray()].astype(DTYPE),
            'active': band.GetMaskBand().ReadAsArray(),
            'geo_transform': dataset.GetGeoTransform(),
            'projection': dataset.GetProjection(),
            'no_data_value': no_data_value,
        }

    def _calculate_hydr_cond(self, datasets):
        # short keys
        i = self.I_SOIL
        o = self.O_HYDRAULIC_CONDUCTIVITY

        # read and convert
        no_data_value = self.no_data_value[o]
        conversion = np.array([no_data_value if x is None else x
                               for x in self.soil_tables['intr_perm']])
        dataset = datasets[i]
        band = dataset.GetRasterBand(1)

        return {
            'values': conversion[band.ReadAsArray()].astype(DTYPE),
            'active': band.GetMaskBand().ReadAsArray(),
            'geo_transform': dataset.GetGeoTransform(),
            'projection': dataset.GetProjection(),
            'no_data_value': no_data_value,
        }

    def calculate(self, datasets):
        """ Return dictionary of output datasets. """
        # Temporarily change no data into 22 in landuse
        use_dataset = datasets[self.I_LANDUSE]
        use_band = use_dataset.GetRasterBand(1)
        use_array = use_band.ReadAsArray()
        use_no_data_value = use_band.GetNoDataValue()
        use_array[use_array == use_no_data_value] = 22
        use_band.WriteArray(use_array)

        # Temporarily change no data into 14 in soil
        soil_dataset = datasets[self.I_SOIL]
        soil_band = soil_dataset.GetRasterBand(1)
        soil_array = soil_band.ReadAsArray()
        soil_no_data_value = soil_band.GetNoDataValue()
        soil_array[soil_array == soil_no_data_value] = 14
        soil_band.WriteArray(soil_array)

        return {key: self.calculators[key](datasets) for key in self.outputs}


class ThreeDiAHN2(Operation, ThreeDiBase):
    name = '3di-ahn2'

    def __init__(self, floor, **kwargs):
        """ Initialize the operation. """
        # first define the layers
        self.layers = {
            self.I_BATHYMETRY: dict(layers=','.join([
                'intern:nl:ahn2:int',
                'intern:nl:ahn2:bag!{}'.format(floor),
            ])),
            self.I_LANDUSE: dict(layers='intern:nl:cover:phy-1801c'),
            self.I_SOIL: dict(layers='soil:3di-1801c'),
        }
        # then let the base class handle the rest
        super(ThreeDiAHN2, self).__init__(floor, **kwargs)


class ThreeDiAHN3(Operation, ThreeDiBase):
    name = '3di-ahn3'

    def __init__(self, floor, **kwargs):
        """ Initialize the operation. """
        # first define the layers
        self.layers = {
            self.I_BATHYMETRY: dict(layers=','.join([
                'intern:nl:ahn3:int-2018',
                'intern:nl:ahn3:bag-2018!{}'.format(floor),
            ])),
            self.I_LANDUSE: dict(layers='intern:nl:cover:phy-1801c'),
            self.I_SOIL: dict(layers='soil:3di-1801c'),
        }
        # then let the base class handle the rest
        super(ThreeDiAHN3, self).__init__(floor, **kwargs)


class ThreeDiRD(Operation, ThreeDiBase):
    name = '3di-rd'

    def __init__(self, floor, **kwargs):
        """ Initialize the operation. """
        # first define the layers
        self.layers = {
            self.I_BATHYMETRY: dict(layers=','.join([
                'intern:nl:rd:int',
                'intern:nl:rd:bag!{}'.format(floor),
            ])),
            self.I_LANDUSE: dict(layers='intern:nl:cover:phy-1801c'),
            self.I_SOIL: dict(layers='soil:3di-1801c'),
        }
        # then let the base class handle the rest
        super(ThreeDiRD, self).__init__(floor, **kwargs)


class Preparation:
    """
    Preparation.
    """
    def __init__(self, path, layer, feature, fields_as_meta, **kwargs):
        """ Prepare a lot. """
        attribute = kwargs.pop('attribute')

        self.server = kwargs.pop('server')
        self.operation = operations[kwargs.pop('operation')](**kwargs)

        self.projection = kwargs.pop('projection')
        self.cellsize = kwargs.pop('cellsize')

        self.wkt = osr.GetUserInputAsWKT(self.projection)
        self.sr = osr.SpatialReference(self.wkt)
        self.paths = self._make_paths(path, layer, feature, attribute)
        self.rpath = self.paths.pop('rpath')
        self.geometry = self._prepare_geometry(feature)

        self.meta = feature.items() if fields_as_meta else None
        self.datasets = self._get_or_create_datasets()

        # Get resume value
        try:
            with open(self.rpath) as resume_file:
                self.resume = int(resume_file.read())
        except IOError:
            self.resume = 0

        self.index = self._create_index()

        if self.index:
            if self.resume > 0:
                print('Resuming from tile {}.'.format(self.resume))
        else:
            print('Already complete.')
            raise CompleteError()

    def _make_paths(self, path, layer, feature, attribute):
        """ Prepare paths from feature attribute or id. """
        try:
            model = feature[attribute]
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

    def _create_dataset(self, name, path):
        """ Prepare output tif dataset. """
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

        # meta
        if self.meta:
            meta = self.meta
        else:
            meta = {n.upper(): self.operation.inputs[n]['layers']
                    for n in self.operation.outputs[name]}
        dataset.SetMetadata({k: str(v) for k, v in meta.items()})
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
        return Index(
            dataset=next(iter(self.datasets.values())),
            geometry=self.geometry,
            resume=self.resume,
        )

    def get_source(self):
        """ Return dictionary of source objects. """
        inputs = self.operation.inputs
        source = Source(index=self.index,
                        server=self.server,
                        projection=self.projection,
                        times={k: inputs[k]['time'] for k in inputs},
                        layers={k: inputs[k]['layers'] for k in inputs})
        return source

    def get_target(self, source):
        """ Return target object. """
        target = Target(source=source,
                        rpath=self.rpath,
                        index=self.index,
                        datasets=self.datasets,
                        geometry=self.geometry,
                        operation=self.operation)
        return target


class Index:
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
        layer = datasource.CreateLayer('geometry', sr)
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
        self.resume = resume
        self.block_size = w, h
        self.dataset_size = dataset.RasterXSize, dataset.RasterYSize
        self.geo_transform = dataset.GetGeoTransform()
        self.indices = index.ReadAsArray().nonzero()

    def get_indices(self, serial):
        """ Return indices into dataset. """
        w, h = self.block_size
        W, H = self.dataset_size
        y, x = self.indices[0][serial].item(), self.indices[1][serial].item()
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
        return POLYGON.format(x1=x1, y1=y1, x2=x2, y2=y2)

    def __len__(self):
        return len(self.indices[0])

    def __bool__(self):
        return len(self) > self.resume

    def __iter__(self):
        for serial in range(self.resume, len(self)):
            x1, y1, x2, y2 = indices = self.get_indices(serial)
            width, height, origin = x2 - x1, y2 - y1, (x1, y1)
            polygon = self.get_polygon(indices)
            yield Tile(width=width,
                       height=height,
                       origin=origin,
                       serial=serial,
                       polygon=polygon)


class Source:
    """
    Factory of source chunks.
    """
    def __init__(self, projection, server, layers, times, index):
        """  """
        self.projection = projection
        self.layers = layers
        self.server = server
        self.times = times
        self.index = index

    def make_url(self, layer, time):
        """ Build the general part of the url. """
        parameters = {'time': time,
                      'layers': layer,
                      'sr': self.projection,
                      'compress': 'deflate',
                      'request': 'getgeotiff'}
        return '{path}?{pars}'.format(
            path=urljoin(self.server, 'data'),
            pars=urlencode(parameters),
        )

    def get_chunks(self, block):
        """ Return dictionary of chunks for a polygon. """
        chunks = {}
        for name in self.layers:
            url = self.make_url(time=self.times[name],
                                layer=self.layers[name])
            chunks[name] = Chunk(url=url, name=name, block=block)
        return chunks


class Chunk():
    """
    Represents a remote chunk of data.
    """
    def __init__(self, url, name, block):
        """ Prepare url. """
        parameters = {'geom': block.tile.polygon,
                      'width': block.tile.width,
                      'height': block.tile.height}
        self.url = '{}&{}'.format(url, urlencode(parameters))
        self.key = '{}_{}'.format(name, block.tile.serial)
        self.name = name
        self.block = block

    def load(self):
        """ Load url into gdal dataset. """
        # retrieve file into gdal vsimem system
        vsi_path = '/vsimem/{}'.format(self.key)
        vsi_file = gdal.VSIFOpenL(vsi_path, 'w')
        url_file = urlopen(self.url)
        size = int(url_file.info().get('content-length'))
        gdal.VSIFWriteL(url_file.read(), size, 1, vsi_file)
        gdal.VSIFCloseL(vsi_file)

        # copy and remove
        dataset = gdal.Open(vsi_path)
        self.block.inputs[self.name] = DRIVER_GDAL_MEM.CreateCopy('', dataset)
        dataset = None
        gdal.Unlink(vsi_path)


class Target:
    """
    Factory of target blocks
    """
    def __init__(self, rpath, index, source, datasets, geometry, operation):
        self.rpath = rpath
        self.index = index
        self.source = source
        self.datasets = datasets
        self.geometry = geometry
        self.operation = operation

    def __len__(self):
        """ Returns the featurecount. """
        return len(self.index)

    def __iter__(self):
        """ Yields blocks. """
        for tile in self.index:
            block = Block(tile=tile,
                          rpath=self.rpath,
                          source=self.source,
                          datasets=self.datasets,
                          geometry=self.geometry,
                          operation=self.operation)
            yield block


class Block:
    """ Self saving local chunk of data. """
    def __init__(self, tile, rpath, source, datasets, geometry, operation):
        self.tile = tile
        self.rpath = rpath
        self.source = source
        self.datasets = datasets
        self.geometry = geometry
        self.operation = operation
        self.chunks = self.source.get_chunks(self)
        self.inputs = {}

    def __iter__(self):
        """ Yield chunks. """
        for chunk in self.chunks.values():
            yield chunk

    def save(self, fill_zeros):
        """
        Cut out and save block.

        param fill_zeros: Put zeros for no data within geometry.
        """
        outputs = self.operation.calculate(self.inputs)
        for name in outputs:

            # assign to names
            output = outputs[name]
            array = output['values'][np.newaxis]
            active = output['active'][np.newaxis]
            projection = output['projection']
            no_data_value = output['no_data_value']
            geo_transform = output['geo_transform']

            # determine inside pixels
            inside = np.zeros_like(active)
            kwargs = {
                'projection': projection,
                'geo_transform': geo_transform,
            }
            with datasources.Layer(self.geometry) as layer:
                with datasets.Dataset(inside, **kwargs) as dataset:
                    gdal.RasterizeLayer(dataset, [1], layer, burn_values=[255])

            if fill_zeros:
                # set inactive pixels to zero, but outside pixels to no data
                array[np.logical_not(active)] = 0
                array[np.logical_not(inside)] = no_data_value
            else:
                # mask outside or inactive
                array[~np.logical_and(active, inside)] = no_data_value

            # write to target dataset
            tile = self.tile
            kwargs.update(no_data_value=no_data_value)
            with datasets.Dataset(array, **kwargs) as dataset:
                data = dataset.ReadRaster(0, 0, tile.width, tile.height)
                args = tile.origin + (tile.width, tile.height, data)
                self.datasets[name].WriteRaster(*args)

        # uppdate resume file
        with open(self.rpath, 'w') as resume_file:
            resume_file.write(str(self.tile.serial + 1))


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


def filler(queue, batch):
    """ Fill queue with chunks from batch and terminate with None. """
    for chunk in batch:
        thread = threading.Thread(target=chunk.load)
        thread.daemon = True
        thread.start()
        queue.put((chunk, thread))
    queue.put(None)


def extract_model(preparation, fill_zeros):
    """
    Extract for a single feature.
    """
    source = preparation.get_source()
    target = preparation.get_target(source)

    total = len(target)
    gdal.TermProgress_nocb(0)
    batch = (c for b in target for c in b)
    queue = queues.Queue(maxsize=8)
    kwargs = {'queue': queue, 'batch': batch}

    thread1 = threading.Thread(target=filler, kwargs=kwargs)
    thread1.daemon = True
    thread1.start()

    while True:
        # fetch loaded chunks
        try:
            chunk, thread2 = queue.get()
            thread2.join()  # this makes sure the chunk is laoded
        except TypeError:
            break

        # save complete blocks
        if len(chunk.block.chunks) == len(chunk.block.inputs):
            chunk.block.save(fill_zeros)
            gdal.TermProgress_nocb((chunk.block.tile.serial + 1) / total)

    thread1.join()


def check_version():
    """
    Check if this is the highest available version of the script.
    """
    url_file = urlopen(GITHUB_URL)
    lines = url_file.read().decode('utf-8').split('\n')
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


def extract_all(shape_path, target_dir, blue_map, **kwargs):
    """
    Prepare and extract for each feature.
    """
    if kwargs['version']:
        print('Extract script version: {}'.format(VERSION))
        exit()
    check_version()
    datasource = ogr.Open(shape_path)
    for layer in datasource:
        total = layer.GetFeatureCount()
        for i in range(total):
            feature = layer[i]
            try:
                preparation = Preparation(layer=layer,
                                          feature=feature,
                                          fields_as_meta=blue_map,
                                          path=target_dir, **kwargs)
            except CompleteError:
                continue
            extract_model(preparation, fill_zeros=blue_map)


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
                        default=SERVER,
                        help='Raster server. Default: "{}"'.format(SERVER))
    parser.add_argument('-o', '--operation',
                        default=OPERATION,
                        choices=operations,
                        help='Output mode. Default: "{}"'.format(OPERATION))
    parser.add_argument('-a', '--attribute',
                        default=ATTRIBUTE,
                        help=('Name of attribute that is used to name the '
                              'output files. Default: "{}"').format(ATTRIBUTE))
    parser.add_argument('-f', '--floor',
                        type=float,
                        default=FLOOR,
                        help=('Floor height (3di). Note that using \'nan\' as '
                              'value will result in buildings having no data '
                              'in the resulting dem. '
                              'Default: {}').format(FLOOR))
    parser.add_argument('-c', '--cellsize',
                        nargs=2,
                        type=float,
                        default=CELLSIZE,
                        help='Cellsize. Default: {} {}'.format(*CELLSIZE))
    parser.add_argument('-p', '--projection',
                        default=PROJECTION,
                        help='Projection. Default: "{}"'.format(PROJECTION))
    parser.add_argument('-t', '--timestamp',
                        default=TIMESTAMP, dest='time',
                        help='Timestamp. Default: "{}"'.format(TIMESTAMP))
    parser.add_argument('-tl', '--landuse',
                        help='Path to landuse csv.')
    parser.add_argument('-ts', '--soil',
                        help='Path to soil csv.')
    parser.add_argument('-l', '--layers',
                        help='Layers for layers operation.')
    parser.add_argument('-dt', '--dtype',
                        default=DTYPE,
                        help=('Numpy datatype for layers operation. '
                              'Default: "{}"').format(DTYPE))
    parser.add_argument('-fv', '--fillvalue',
                        help=('Fillvalue for layers operation. '
                              'If not given, the maximum possible '
                              'number of the output dtype will be used.'))
    parser.add_argument('-b', '--blue-map', action='store_true', help=(
        'Enable Blue Map mode. Output metadata is set from the source model '
        'attributes and no data values within the model are set to 0.'
    ))
    return parser


def main():
    """ Call extract_all with args from parser. """
    operations.update({cls.name: cls for cls in Operation.__subclasses__()})
    return extract_all(**vars(get_parser().parse_args()))


if __name__ == '__main__':
    exit(main())
