# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.
"""
Rextract, the king of extractors.

Extract parts of lizard rasters using geometries from a shapefile.

Please note that any information about the spatial reference system in the
shapefile is ignored.

If something goes wrong due to a problem on one of the lizard servers, it may
be possible to resume the process by keeping the output folder intact and
retrying exactly the same command.
"""

import argparse
import contextlib
import getpass
import os
import pathlib
import requests
import tempfile
import queue as queues
import threading

import numpy as np

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

from raster_tools import datasources
from raster_tools import datasets

# urls and the like
USER_AGENT = 'Nelen-Schuurmans/Raster-Tools/Rextract'
API_URL = 'https://nxt3.staging.lizard.net/api/v4/rasters/'
LOGIN_URL = 'https://nxt3.staging.lizard.net/api-auth/login/'
GITHUB_URL = (
    'https://raw.github.com/nens/raster-tools/master/raster_tools/rextract.py'
)

# gdal drivers and options
MEM_DRIVER = gdal.GetDriverByName('mem')
TIF_DRIVER = gdal.GetDriverByName('gtiff')
TIF_OPTIONS = [
    'TILED=YES',
    'BIGTIFF=YES',
    'SPARSE_OK=TRUE',
    'COMPRESS=DEFLATE',
]

# dtype argument lookups
DTYPES = {'u1': gdal.GDT_Byte,
          'u2': gdal.GDT_UInt16,
          'u4': gdal.GDT_UInt32,
          'i2': gdal.GDT_Int16,
          'i4': gdal.GDT_Int32,
          'f4': gdal.GDT_Float32}

# polygon template
POLYGON = 'POLYGON (({x1} {y1},{x2} {y1},{x2} {y2},{x1} {y2},{x1} {y1}))'

# argument defaults
TIMESTAMP = '1970-01-01T00:00:00Z'
ATTRIBUTE = 'name'
SRS = 'EPSG:28992'
CELLSIZE = 0.5
DTYPE = 'f4'

# replace this with dataclass in python 3.7


class Indicator:
    def __init__(self, path):
        self.path = str(path)

    def get(self):
        try:
            with open(self.path) as f:
                return int(f.read())
        except IOError:
            return 0

    def set(self, value):
        with open(self.path, 'w') as f:
            f.write('%s\n' % value)


class Index:
    """ Iterates the indices into the target dataset. """
    def __init__(self, dataset, geometry):
        """
        Rasterize geometry into target dataset extent to find relevant
        blocks.
        """
        # make a dataset
        w, h = dataset.GetRasterBand(1).GetBlockSize()
        p, a, b, q, c, d = dataset.GetGeoTransform()
        index = MEM_DRIVER.Create(
            '',
            (dataset.RasterXSize - 1) // w + 1,
            (dataset.RasterYSize - 1) // h + 1,
            1,
            gdal.GDT_Byte,
        )

        geo_transform = p, a * w, b * h, q, c * w, d * h
        index.SetProjection(dataset.GetProjection())
        index.SetGeoTransform(geo_transform)

        # find active blocks by rasterizing geometry
        options = ['all_touched=true']
        with datasources.Layer(geometry) as layer:
            gdal.RasterizeLayer(
                index, [1], layer, burn_values=[1], options=options,
            )

        # store as attributes
        self.block_size = w, h
        self.dataset_size = dataset.RasterXSize, dataset.RasterYSize
        self.geo_transform = dataset.GetGeoTransform()
        self.indices = index.ReadAsArray().nonzero()

    def _get_indices(self, serial):
        """ Return indices into dataset. """
        w, h = self.block_size
        W, H = self.dataset_size
        y, x = self.indices[0][serial].item(), self.indices[1][serial].item()
        x1 = w * x
        y1 = h * y
        x2 = min(W, (x + 1) * w)
        y2 = min(H, (y + 1) * h)
        return x1, y1, x2, y2

    def _get_geom(self, indices):
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

    def get_chunks(self, start=1):
        """
        Return chunk generator.

        Note that the serial number starts counting at 1.
        """
        for serial in range(start, len(self) + 1):
            x1, y1, x2, y2 = indices = self._get_indices(serial - 1)
            width, height, origin = x2 - x1, y2 - y1, (x1, y1)
            geom = self._get_geom(indices)
            yield Chunk(
                geom=geom,
                width=width,
                height=height,
                origin=origin,
                serial=serial,
            )


class Target:
    """
    Wraps the resulting gdal dataset.
    """
    def __init__(self, path, geometry, dtype, fillvalue, **kwargs):
        """ Kwargs contain cellsize and uuid. """
        # coordinates
        self.geometry = geometry

        # types
        self.dtype = dtype
        if fillvalue is None:
            info = np.finfo if dtype.startswith('f') else np.iinfo
            self.fillvalue = info(dtype).max.item()
        else:
            self.fillvalue = fillvalue

        # dataset
        if path.exists():
            print('Appending to %s... ' % path, end='')
            self.dataset = gdal.Open(str(path), gdal.GA_Update)
        else:
            print('Creating %s' % path)
            self.dataset = self._create_dataset(path=str(path), **kwargs)

        # chunks
        self.index = Index(dataset=self.dataset, geometry=self.geometry)

    def __len__(self):
        return len(self.index)

    @property
    def data_type(self):
        return DTYPES[self.dtype]

    @property
    def no_data_value(self):
        return self.fillvalue

    @property
    def projection(self):
        return self.geometry.GetSpatialReference().ExportToWkt()

    def _create_dataset(self, path, cellsize, time, uuid):
        """ Create output tif dataset. """
        # calculate
        a, b, c, d = cellsize, 0.0, 0.0, -cellsize
        x1, x2, y1, y2 = self.geometry.GetEnvelope()
        p, q = a * (x1 // a), d * (y2 // d)

        width = -int((p - x2) // a)
        height = -int((q - y1) // d)
        geo_transform = p, a, b, q, c, d

        # create
        dataset = TIF_DRIVER.Create(
            path, width, height, 1, self.data_type, options=TIF_OPTIONS,
        )
        dataset.SetProjection(self.projection)
        dataset.SetGeoTransform(geo_transform)
        dataset.GetRasterBand(1).SetNoDataValue(self.no_data_value)

        # meta
        dataset.SetMetadata({'time': time, 'uuid': uuid})

        return dataset

    def get_chunks(self, start):
        return self.index.get_chunks(start)

    def save(self, chunk):
        """
        """
        # read and convert datatype
        with chunk.as_dataset() as dataset:
            band = dataset.GetRasterBand(1)
            active = band.GetMaskBand().ReadAsArray()[np.newaxis]
            array = band.ReadAsArray().astype(self.dtype)[np.newaxis]

        # determine inside pixels
        inside = np.zeros_like(active)
        kwargs = {
            'geo_transform': dataset.GetGeoTransform(),
            'projection': dataset.GetProjection(),
        }
        with datasources.Layer(self.geometry) as layer:
            with datasets.Dataset(inside, **kwargs) as dataset:
                gdal.RasterizeLayer(dataset, [1], layer, burn_values=[255])

        # mask outide or inactive
        array[~np.logical_and(active, inside)] = self.no_data_value

        # write to target dataset
        kwargs.update(no_data_value=self.no_data_value)
        with datasets.Dataset(array, **kwargs) as dataset:
            data = dataset.ReadRaster(0, 0, chunk.width, chunk.height)
            args = chunk.origin + (chunk.width, chunk.height, data)
        self.dataset.WriteRaster(*args)


class Chunk(object):
    def __init__(self, geom, width, height, origin, serial):
        # for request
        self.geom = geom
        self.width = width
        self.height = height

        # for result
        self.origin = origin
        self.serial = serial

        # the geotiff data
        self.response = None

    def fetch(self, session, time, uuid, srs):
        request = {
            'url': API_URL + uuid + '/data/',
            'params': {
                'srs': srs,
                'time': time,
                'geom': self.geom,
                'width': self.width,
                'height': self.height,
                'format': 'geotiff',
                'compress': 'deflate',
            }
        }

        self.response = session.get(**request)

    @contextlib.contextmanager
    def as_dataset(self):
        """ Temporily serve data as geotiff file in virtual memory. """
        _, p = tempfile.mkstemp(dir='/dev/shm')
        try:
            with open(p, 'wb') as f:
                f.write(self.response.content)
            yield gdal.Open(p)
        finally:
            os.remove(p)


def filler(queue, chunks, **kwargs):
    """ Fill queue with chunks from batch and terminate with None. """
    for chunk in chunks:
        thread = threading.Thread(target=chunk.fetch, kwargs=kwargs)
        thread.daemon = True
        thread.start()
        queue.put((thread, chunk))
    queue.put(None)


class RasterExtraction:
    """
    Represent the extraction of a single feature.
    """
    def __init__(self, path, **kwargs):
        self.indicator = Indicator(path=path.with_suffix('.pro'))
        self.target = Target(path=path.with_suffix('.tif'), **kwargs)

    def process(self, session, srs, time, uuid):
        """
        Extract for a single feature.

        The kwargs are passed to the fetch() method of each chunk and should
        define session, time and uuid.
        """
        completed = self.indicator.get()
        total = len(self.target)
        if completed == total:
            print('Already complete.')
            return
        if completed > 0:
            print('Resuming from chunk %s.' % completed)

        gdal.TermProgress_nocb(completed / total)

        # run a thread that starts putting chunks with threads in a queue
        queue = queues.Queue(maxsize=4)
        filler_kwargs = {
            'queue': queue,
            'chunks': self.target.get_chunks(start=completed + 1),
            'session': session,
            'srs': srs,
            'time': time,
            'uuid': uuid,
        }
        filler_thread = threading.Thread(target=filler, kwargs=filler_kwargs)
        filler_thread.daemon = True
        filler_thread.start()

        while True:
            # fetch loaded chunks
            try:
                fetch_thread, chunk = queue.get()
                fetch_thread.join()  # this makes sure the chunk is loaded
            except TypeError:
                self.indicator.set(completed)
                break

            # save complete blocks
            if chunk.response.status_code != 200:
                self.indicator.set(completed)
                raise ValueError('Error retrieving data from lizard.')
            self.target.save(chunk)
            completed = chunk.serial
            gdal.TermProgress_nocb(completed / total)
        filler_thread.join()


def rextract(shape_path, output_path, username, attribute, srs, **kwargs):
    """
    Prepare and extract for each feature.
    """
    # session
    if username is None:
        # no login
        session = requests
    else:
        # login, might be needed for every thread...
        password = getpass('password for %s: ' % username)
        session = requests.Session()
        session.post(
            url=LOGIN_URL,
            data={'username': username, 'password': password},
        )

    # extract
    sr = osr.SpatialReference(osr.GetUserInputAsWKT(srs))
    output_path.mkdir(exist_ok=True)
    for layer in ogr.Open(shape_path):
        layer_path = output_path / layer.GetName()
        layer_path.mkdir(exist_ok=True)
        for feature_no in range(layer.GetFeatureCount()):
            feature = layer[feature_no]
            geometry = feature.geometry()
            geometry.AssignSpatialReference(sr)  # ignore original srs
            raster_extraction = RasterExtraction(
                path=layer_path / feature[attribute],
                geometry=geometry,
                **kwargs,
            )
            raster_extraction.process(
                session=session,
                srs=srs,
                time=kwargs['time'],
                uuid=kwargs['uuid']
            )


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    # main
    parser.add_argument(
        'shape_path',
        metavar='SHAPE',
        help='Shapefile with outlining geometries.'
    )
    parser.add_argument(
        'output_path',
        metavar='OUTPUT',
        type=pathlib.Path,
        help='Directory to place output rasters.'
    )
    parser.add_argument(
        'uuid',
        metavar='UUID',
        help='UUID of the raster to extract.',
    )
    # options
    parser.add_argument(
        '-u', '--username',
        help='Lizard username.',
    )
    parser.add_argument(
        '-a', '--attribute', default=ATTRIBUTE,
        help=(
            'Shapefile attribute for naming result files. Default: "%s"'
        ) % ATTRIBUTE,
    )
    parser.add_argument(
        '-c', '--cellsize', nargs=2, type=float, default=CELLSIZE,
        help='Cellsize. Default: %s' % CELLSIZE,
    )
    parser.add_argument(
        '-s', '--srs', default=SRS,
        help='Spatial reference system. Default: "%s"' % SRS,
    )
    parser.add_argument(
        '-t', '--timestamp',
        default=TIMESTAMP, dest='time',
        help='Timestamp. Default: "%s"' % TIMESTAMP,
    )
    parser.add_argument(
        '-d', '--dtype', default=DTYPE, choices=DTYPES,
        help='Numpy datatype for resulting rasters. Default: "%s"' % DTYPE,
    )
    parser.add_argument(
        '-f', '--fillvalue',
        help=(
            'No data value for resulting rasters. Defaults to the maximum '
            'possible value of the output datatype.'
        )
    )
    return parser


def main():
    """ Call command with args from parser. """
    rextract(**vars(get_parser().parse_args()))
