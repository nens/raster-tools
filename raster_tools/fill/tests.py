# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-

import sys
import unittest

from osgeo import gdal
from scipy import ndimage
import numpy as np

from raster_tools.fill import fill
from raster_tools.fill import edges
from raster_tools import datasets


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

        # save a geotiff in memory
        driver = gdal.GetDriverByName('gtiff')
        cls.path = '/vsimem/source.tif'
        array = source[np.newaxis]
        with datasets.Dataset(array, no_data_value=fillvalue) as dataset:
            driver.CreateCopy(cls.path, dataset)

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
        sys.argv = ['', self.path, '/vsimem/target.tif']

        # run
        fill.main()  # run, but cannot find source (because it is in memory)

        # patch exists
        exists = fill.exists
        fill.exists = gdal.VSIStatL

        fill.progress = False  # suppress the progress bar
        fill.main()            # finds and fills the in-memory dataset
        fill.main()            # skips because the target exists

        # restore
        sys.argv = argv
        fill.exists = exists
