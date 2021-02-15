# -*- coding: utf-8 -*-

# (c) Nelen & Schuurmans, see LICENSE.rst.
"""
Vectorize flow.
"""

import os

import numpy as np

from raster_tools import groups
from raster_tools import datasources

from raster_tools import gdal
from raster_tools import ogr
from raster_tools import osr

SHAPE = ogr.GetDriverByName(str('esri shapefile'))
COURSES = np.array([(64, 128, 1),
                    (32, 0, 2),
                    (16, 8, 4)], 'u1')

INDICES = COURSES.nonzero()
NUMBERS = COURSES[INDICES][np.newaxis, ...]
OFFSETS = np.array(INDICES).transpose() - 1

CLASSES = ((2.0, 3.0),
           (3.0, 4.0),
           (4.0, 4.7),
           (4.7, 9.9))


def get_traveled(courses):
    """ Return indices when travelling along courses. """
    # turn indices into points array
    height, width = courses.shape
    indices = (np.arange(height).repeat(width),
               np.tile(np.arange(width), height))
    points = np.array(indices).transpose()

    # determine direction and apply offset
    encode = courses[indices][:, np.newaxis]     # which codes
    select = np.bool8(encode & NUMBERS)          # which courses
    target = points + OFFSETS[select.argmax(1)]  # apply offsets

    return tuple(target.transpose())             # return tuple


def vectorize(direction, accumulation):
    """
    Vectorize flow.

    Key principle is the flow array, that relates the source cell A to
    the target cell B as B = flow[A].

    A special value equal to size indicates the stream leaving the flow.
    """
    # construct a mapping array for the flow
    size = direction.size
    height, width = direction.shape
    traveled = get_traveled(direction)

    # construct the flow array
    flow = np.empty(size + 1, dtype='i8')
    flow[-1] = size
    flow[:size] = np.where(np.logical_or.reduce([
        direction.ravel() == 0,    # undefined cells
        traveled[0] < 0,        # flow-off to the top
        traveled[0] >= height,  # ... bottom
        traveled[1] < 0,        # ... left
        traveled[1] >= width,   # ... right
    ]), size, traveled[0] * width + traveled[1])

    # eliminate opposing directions
    state = np.arange(size)
    flow[:-1][flow[flow[state]] == state] = size

    for lower, upper in CLASSES:
        # select points that match klass
        points = (np.logical_and(accumulation.ravel() < upper,
                                 accumulation.ravel() >= lower)).nonzero()[0]

        # determine sources, merges and sinks
        flowed = flow[points]
        leaving = (flowed == size)
        promoting = np.logical_and(~leaving,
                                   np.in1d(flowed, points, invert=True))
        bincount = np.bincount(flowed, minlength=size)[:-1]

        sources = points[np.logical_and(
            flowed != size,
            np.in1d(points, flowed, invert=True),
        )]
        merges = np.intersect1d(points, np.where(bincount > 1)[0])
        sinks = np.union1d(points[leaving], flowed[promoting])

        # determine starts and stops
        starts = np.union1d(sources, merges)
        stops = set(np.union1d(merges, sinks).tolist())  # native set

        # travel them and yield per section
        for x in starts:
            if x in sinks:
                continue
            line = [x]
            while True:
                x = flow[x]
                line.append(x)
                if x in stops:
                    break
            a = np.array(line)
            yield lower, (a // width - 0.5, a % width - 0.5)  # pixel center


class Vectorizer(object):
    def __init__(self, direction_path, accumulation_path, target_path):
        # paths and source data
        self.direction_group = groups.Group(gdal.Open(direction_path))
        self.accumulation_group = groups.Group(gdal.Open(accumulation_path))
        self.target_path = target_path

        # geospatial reference
        self.geo_transform = self.direction_group.geo_transform
        self.projection = self.direction_group.projection

    def vectorize(self, index_feature):
        # target path
        name = index_feature[str('name')]
        path = os.path.join(self.target_path, name[:3], '{}'.format(name))
        if os.path.exists(path):
            return

        # create directory
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass  # no problem

        index_geometry = index_feature.geometry()
        geo_transform = self.geo_transform.shifted(index_geometry)

        # data with one pixel margin on all sides
        indices = self.geo_transform.get_indices(index_geometry)
        indices = (indices[0] - 1,
                   indices[1] - 1,
                   indices[2] + 1,
                   indices[3] + 1)
        direction = self.direction_group.read(indices)
        accumulation = self.accumulation_group.read(indices)

        # processing
        data_source = SHAPE.CreateDataSource(str(path))
        layer_sr = osr.SpatialReference(self.projection)
        layer_name = str(os.path.basename(path))
        layer = data_source.CreateLayer(layer_name, layer_sr)
        layer.CreateField(ogr.FieldDefn(str('class'), ogr.OFTReal))
        layer_defn = layer.GetLayerDefn()
        generator = vectorize(direction=direction, accumulation=accumulation)
        for klass, indices in generator:
            feature = ogr.Feature(layer_defn)
            points = geo_transform.get_coordinates(indices)
            feature[str('class')] = klass
            geometry = ogr.Geometry(ogr.wkbLineString)
            for p in zip(*points):
                geometry.AddPoint_2D(*p)
            feature.SetGeometry(geometry)
            layer.CreateFeature(feature)


def flow_vec(index_path, part, **kwargs):
    """
    """
    # select some or all polygons
    index = datasources.PartialDataSource(index_path)
    if part is not None:
        index = index.select(part)

    vectorizer = Vectorizer(**kwargs)

    for feature in index:
        vectorizer.vectorize(feature)
    return 0


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        'index_path',
        metavar='INDEX',
        help='shapefile with geometries and names of output tiles',
    )
    parser.add_argument(
        'direction_path',
        metavar='DIRECTION',
        help='GDAL direction raster dataset',
    )
    parser.add_argument(
        'accumulation_path',
        metavar='ACCUMULATION',
        help='GDAL accumulation raster dataset',
    )
    parser.add_argument(
        'target_path',
        metavar='OUTPUT',
        help='target folder',
    )
    parser.add_argument(
        '-p', '--part',
        help='partial processing source, for example "2/3"',
    )
    return parser


def main():
    """ Call aggregate with args from parser. """
    kwargs = vars(get_parser().parse_args())
    flow_vec(**kwargs)
