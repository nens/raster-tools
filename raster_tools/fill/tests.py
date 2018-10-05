# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-

import sys
import unittest

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from scipy import ndimage
import numpy as np

from raster_tools.fill import fill
from raster_tools.fill import edges
from raster_tools import datasets
from raster_tools import datasources

POLYGON = 'POLYGON (({x1} {y1},{x2} {y1},{x2} {y2},{x1} {y2},{x1} {y1}))'


class TestFillNoData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # create source data
        dtype = 'f4'
        fillvalue = np.finfo(dtype).max.item()
        shape = 7, 7
        sample = sum(np.indices(shape)).astype('f4')
        cls.void = ndimage.binary_erosion(np.ones(shape, dtype='b1'))
        source = np.where(cls.void, fillvalue, sample)

        # create an edge
        edge = cls.void ^ ndimage.binary_dilation(cls.void)
        indices = edge.nonzero()
        cls.edge = edges.Edge(
            indices=indices,
            values=source[indices],
            shape=source.shape,
        )

        # save a raster in memory
        driver = gdal.GetDriverByName('Gtiff')
        cls.raster = '/vsimem/raster.tif'
        array = source[np.newaxis]
        projection = osr.GetUserInputAsWKT('EPSG:28992')
        x1, y2 = 200000, 400007
        kwargs = {
            'geo_transform': (x1, 1, 0, y2, 0, -1),
            'no_data_value': fillvalue,
            'projection': projection,
        }
        with datasets.Dataset(array, **kwargs) as dataset:
            driver.CreateCopy(cls.raster, dataset)

        # save a shape in memory
        driver = ogr.GetDriverByName('ESRI Shapefile')
        cls.vector = '/vsimem/vector.shp'
        data_source = driver.CreateDataSource(cls.vector)
        y1, x2 = 400000, 200007
        wkt = POLYGON.format(x1=x1, y1=y1, x2=x2, y2=y2)
        sr = osr.SpatialReference(projection)
        geometry = ogr.CreateGeometryFromWkt(wkt, sr)
        with datasources.Layer(geometry) as layer:
            data_source.CopyLayer(layer, '')

    def test_edge(self):
        self.assertEqual(
            7.0,
            self.edge.aggregated().aggregated().aggregated().toarray().item(),
        )

    def test_fill(self):
        filled = fill.fill(self.edge)
        self.assertAlmostEqual(
            6.0,
            filled[self.void].mean(),
            places=0,
        )

    def test_script(self):
        # patch
        argv = sys.argv
        sys.argv = [
            '',
            self.raster,
            '/vsimem/output.tif',
            '--clip', '/a/bogus/path',
            '--round', '2',
        ]

        # run but will not find the source raster, because it is in memory
        fill.main()

        # patch exists
        exists = fill.exists
        fill.exists = gdal.VSIStatL

        # run and fail to find the clip source on the bogus path
        fill.main()

        sys.argv[4] = self.vector  # supply correct clip source path
        fill.progress = False      # suppress the progress bar
        fill.main()                # finds and fills the in-memory dataset
        fill.main()                # skips because the target exists

        # restore
        sys.argv = argv
        fill.exists = exists
