"""
Fill holes, then draw a contour around the raster.

Usage:

python contour.py source.tif target.gpkg
"""

from argparse import ArgumentParser
# from functools import partial
from pathlib import Path

from osgeo.gdal import (
    ContourGenerateEx,
    # GetDriverByName as GetRasterDriverByName,
    Open,
)
from osgeo.gdal_array import OpenArray
from osgeo.ogr import (
    FieldDefn,
    GetDriverByName as GetVectorDriverByName,
    OFTInteger,
    OFTReal,
)
from osgeo.osr import SpatialReference
from scipy.ndimage import binary_fill_holes

DRIVER_GPKG = GetVectorDriverByName('GPKG')
# DRIVER_GTIF = GetRasterDriverByName('GTiff')

PAD = 10


def contour(source_path, target_path):

    dataset = Open(str(source_path))
    band = dataset.GetRasterBand(1)
    values = dataset.ReadAsArray()
    no_data_value = band.GetNoDataValue()
    array = (values != no_data_value).astype('u1')

    src = OpenArray(array, prototype_ds=dataset)
    # prepare to dump some tiffs
    # options = ['compress=deflate']
    # dump = partial(DRIVER_GTIF.CreateCopy, src=src, options=options)
    # dump('results/original.tif')

    # hole filled
    binary_fill_holes(array, output=array)
    # dump('results/fill_holes.tif')

    # write contour
    srs = SpatialReference(dataset.GetProjection())
    data_source = DRIVER_GPKG.CreateDataSource(str(target_path))
    layer = data_source.CreateLayer('contour', srs=srs)
    layer.CreateField(FieldDefn('id', OFTInteger))
    layer.CreateField(FieldDefn('elev_min', OFTReal))
    layer.CreateField(FieldDefn('elev_max', OFTReal))
    ContourGenerateEx(
        src.GetRasterBand(1),
        layer,
        options=[
            'FIXED_LEVELS=0.5',
            'ELEV_FIELD_MIN=1',
            'ELEV_FIELD_MAX=2',
            'POLYGONIZE=TRUE',
        ],
    )
    layer.DeleteFeature(1)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        'source_path',
        metavar='RASTER',
        type=Path,
        help='source raster file'
    )
    parser.add_argument(
        'target_path',
        metavar='OUTPUT',
        type=Path,
        help='output vector geopackage',
    )
    args = parser.parse_args()
    contour(source_path=args.source_path, target_path=args.target_path)
