# -*- coding: utf-8 -*-
"""
Find lowest upstream points along a line within a polygon using
combined data from raster stores.
"""

import argparse
import math

from osgeo import gdal
from osgeo import ogr
import numpy as np

from raster_tools import datasources
from raster_tools import groups


POINT = 'POINT({} {})'
KEY = 'height'


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        'polygon_path',
        metavar='POLYGONS',
        help='Confine search to these polygons.',
    )
    parser.add_argument(
        'linestring_path',
        metavar='LINES',
        help='Assign height to points on these lines.',
    )
    parser.add_argument(
        'raster_paths',
        metavar='STORE',
        nargs='+',
        help=('Get raster data from this raster '
              'store (multiple stores possible).'),
    )
    parser.add_argument(
        'path',
        metavar='POINTS',
        help='Path to output (point-)shapefile',
    )
    parser.add_argument(
        '-g', '--grow',
        type=float,
        default=0.5,
        metavar='',
        help='Initial buffer of input polygons (default 0.5).',
    )
    parser.add_argument(
        '-d', '--distance',
        type=float,
        default=15.0,
        metavar='',
        help='Minimum upstream search distance (default 15.0).',
    )
    parser.add_argument(
        '-m', '--multiplier',
        type=float,
        default=1.0,
        metavar='',
        help='Multiplier for distance to polygon (default 1.0).',
    )
    parser.add_argument(
        '-s', '--separation',
        type=float,
        default=1.0,
        metavar='',
        help='Separation between points (default 1.0)',
    )
    parser.add_argument(
        '-p', '--partial',
        help='Partial processing source, for example "2/3"',
    )
    return parser


def point2geometry(point, sr):
    """ Return ogr geometry. """
    return ogr.CreateGeometryFromWkt(POINT.format(*point), sr)


class MinimumGroup(object):
    def __init__(self, paths):
        self.groups = [groups.Group(gdal.Open(path)) for path in paths]

    def read(self, bounds):
        """
        Return single raster which is the elementwise miminim of the source
        rasters.
        """
        values = np.array([group.read(bounds) for group in self.groups])
        no_data_values = np.array([group.no_data_value
                                   for group in self.groups])

        # replace the no data values with the dtypes' maximum
        dtype = values.dtype
        maximum = {'f': np.finfo, 'i': np.info}[dtype.kind](dtype).max
        values[values == no_data_values] = maximum

        return {'values': values.min(0), 'no_data_value': maximum}


class Case(object):
    def __init__(self, group, polygon, distance,
                 multiplier, separation, linestring):
        self.group = group
        self.polygon = polygon
        self.distance = distance
        self.multiplier = multiplier
        self.linestring = linestring
        self.separation = separation
        self.sr = linestring.GetSpatialReference()

    def get_pairs(self, reverse):
        """ Return generator of point pairs. """
        linestring = self.linestring.Clone()
        linestring.Segmentize(self.separation)
        points = linestring.GetPoints()
        if reverse:
            points.reverse()
        return zip(points[:-1], points[1:])

    def get_sites(self, reverse):
        """ Return generator of geometry, normal pairs. """
        for (x1, y1), (x2, y2) in self.get_pairs(reverse):
            dx, dy = x2 - x1, y2 - y1
            magnitude = math.sqrt(dx ** 2 + dy ** 2)
            direction = dx / magnitude, dy / magnitude
            yield point2geometry((x1, y1), self.sr), direction
        # yield the last point with previous direction
        yield point2geometry((x2, y2), self.sr), direction

    def make_rectangle(self, point, radius, direction):
        """ Return ogr geometry of rectangle. """
        sr = point.GetSpatialReference()
        x, y = point.GetPoints()[0]
        points = []
        dx, dy = direction
        dx, dy = 2 * dx * radius, 2 * dy * radius  # scale
        dx, dy = dy, -dx  # right
        x, y = x + dx, y + dy  # move
        points.append('{} {}'.format(x, y))
        dx, dy = -dy, dx  # left
        x, y = x + dx, y + dy  # move
        points.append('{} {}'.format(x, y))
        dx, dy = -dy, dx  # left
        x, y = x + dx, y + dy  # move
        x, y = x + dx, y + dy  # move
        points.append('{} {}'.format(x, y))
        dx, dy = -dy, dx  # left
        x, y = x + dx, y + dy  # move
        points.append('{} {}'.format(x, y))
        points.append(points[0])
        wkt = 'POLYGON ((' + ','.join(points) + '))'
        return ogr.CreateGeometryFromWkt(wkt, sr)

    def get_areas(self, reverse):
        """ Return generator of point, area tuples. """
        for point, direction in self.get_sites(reverse):
            if not self.polygon.Contains(point):
                continue
            radius = max(
                self.distance,
                self.multiplier * point.Distance(self.polygon.Boundary()),
            )
            circle = point.Buffer(radius)
            rectangle = self.make_rectangle(point=point,
                                            radius=radius,
                                            direction=direction)
            intersection = circle.Intersection(rectangle)

            yield point, self.polygon.Intersection(intersection)

    def get_levels(self, reverse):
        """ Return generator point, level tuples. """
        for point, polygon in self.get_areas(reverse):

            if polygon.GetGeometryName() == 'MULTIPOLYGON':
                # keep reference to original collection or segfault
                collection = polygon
                polygon = min(collection, key=point.Distance)
                polygon.AssignSpatialReference(
                    collection.GetSpatialReference(),
                )

            # get data from store
            data = self.group.read(bounds=polygon)
            values = data['values']
            array = values[values != data['no_data_value']]
            if not array.size:
                continue
            level = array[array.argsort()[1]].item()
            # print(level)
            # if level < -4.8:
            #     from raster_analysis import plots
            #     plot = plots.Plot()
            #     ma = np.ma.masked_equal(data['values'],
            #                             data['no_data_value'])
            #     plot.add_array(ma[0], extent=polygon.GetEnvelope())
            #     #plot.add_geometries(point, polygon, self.polygon)
            #     plot.add_geometries(point, polygon)
            #     plot.show()
            yield point, level


def upstream(polygon_path, linestring_path, raster_paths,
             grow, distance, multiplier, separation, path, partial):
    # open files
    linestring_features = datasources.PartialDataSource(linestring_path)
    group = MinimumGroup(raster_paths)
    target = datasources.TargetDataSource(
        path=path,
        template_path=linestring_path,
        attributes=[KEY],
    )

    # select some or all polygons
    source = datasources.PartialDataSource(polygon_path)
    if partial is None:
        polygon_features = source
    else:
        polygon_features = source.select(partial)

    for polygon_feature in polygon_features:
        # grow a little
        polygon = polygon_feature.geometry().Buffer(grow)

        # query the linestrings
        for linestring_feature in linestring_features.query(polygon):
            linestring = linestring_feature.geometry()

            case = Case(group=group,
                        polygon=polygon,
                        distance=distance,
                        multiplier=multiplier,
                        separation=separation,
                        linestring=linestring)

            # do
            points, levels = zip(*list(case.get_levels(False)))

            if len(levels) > 1:
                # check upstream
                index = int(len(levels) / 2)
                first = levels[:index]
                last = levels[index:]
                if sum(first) / len(first) > sum(last) / len(last):
                    # do reverse
                    try:
                        points, levels = zip(*list(case.get_levels(True)))
                    except (TypeError, ValueError):
                        # there are no levels for this case
                        continue

            # save
            attributes = dict(linestring_feature.items())
            for point, level in zip(points, levels):
                attributes[KEY] = level
                target.append(geometry=point, attributes=attributes)
    return 0


def main():
    """ Call upstream with args from parser. """
    return upstream(**vars(get_parser().parse_args()))
