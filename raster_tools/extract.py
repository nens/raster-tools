# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import json
import logging
import sys
import urllib

from osgeo import gdal
from osgeo import ogr

logger = logging.getLogger(__name__)


class Operation(object):
    """ Base class for operations. """


class Elevation(Operation):
    name = 'elevation'
    store_layers = 'elevation'
    data_type = 6
    no_data_value = 3.4028235e+38
    
    @classmethod
    def calculate(cls, arrays, **kwargs):
        pass


operations = {cls.name: cls for cls in Operation.__subclasses__()}


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
                        default='raster.lizard.net')
    parser.add_argument('-o', '--operation',
                        default='elevation',
                        help='Operation')
    parser.add_argument('-a', '--attribute',
                        help='Attribute for tif filename.')
    parser.add_argument('-f', '--floor',
                        default=0.15,
                        help='Floor elevation above ground level')
    parser.add_argument('-c', '--cellsize',
                        default=(0.5, 0.5),
                        help='Cellsize for output file')
    parser.add_argument('-p', '--projection',
                        default='epsg:28992',
                        help='Spatial reference system for output file.')
    return parser


def get_remote_index(server, layer, feature, projection):
    """
localhost:5000/data?request=getstrategy&layers=elevation&polygon=POLYGON((0.1%200.1,255.9%200.1,%20255.9%20255.9,%200.1%20255.9,0.1%200.1))&projection=epsg:28992&width=256&height=256
    """
    """
    Return ogr memory datasource
    """
    ogr.CreateGeometryFromWkt('POLYGON((0 0,0 1,1 0,0 0)))').Area()
    import ipdb
    ipdb.set_trace() 
    polygon = layer.GetExtent()
    # build the strayegy request url
    get_parameters = dict(
        request='getstrategy',
        layers=layers,
        polygon=None,
    )
    remote = urllib.urlopen(url)
    strategy = None
    ogr_driver = ogr.GetDriverByName('Memory')
    return json.load(remote)

def create_gdal_dataset(strategy, target_dir, name):
    """ Create the big tiff dataset. """
    gdal_driver = gdal.GetDriverByName('gtiff')
    gdal_driver.Create()


def get_dataset(layer, block):
    pass


def get_dataset_from_server(dictionary):
    """ Fetch a gdal dataset from a layer on a raster-server. """
    # build the url
    url = urllib.urlencode(dict())
    get_parameters = dict(
        polygon=dictionary['polygon'].ExportToWkt(),
        projection=dictionary['projection'],
        width=str(dictionary['width']),
        height=str(dictionary['height']),
        compress=dictionary.get('compress', 'deflate')
    )
    get_parameters
    exit()
    url_file = urllib.urlopen(url)
    vsi_file = gdal.VSIFOpenL('myfile', 'w')
    vsi_file.write(url_file.read())
    vsi_file.close()


def command(server, ogr_path, operation):
    """
    Need remote transports based on the strategy, cached
    Need local transports being given to operation
    operation.calculate()
    operation.result()
    tif.put(operation.result)


        loadcontainer objects that can be queued for filling with data
    Need container objects that can be 


    6. Per feature:
        1. Ask for global strategy for all layers and the extent of the shape
        a. Create geotiff based on rounded ahn2 resolution
        b. Create local index
        c. Some code
            for local block:
                read local block as dataset
                for (cached) remote block:
                    for layer:
                        warp to local-block-version of dataset
                do optional modifications
                write local block
                update metadata completeness counter
    """
    ogr_datasource = ogr.Open(shape_path)
    ogr_layer = ogr_datasource[0]
    
    strategy = get_strategy(server=server,
                            ogr_layer=ogr_layer,
                            raster_layers=raster_layers)

    gdal_dataset = create_gdal_dataset(gtiff_path, strategy)


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return command(**vars(get_parser().parse_args()))



