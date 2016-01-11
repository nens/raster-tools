# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

""" Convert a shapefile containing 2D linestrings to a shapefile with
embedded elevation from an elevation map.

Target shapefile can have two layouts: A 'point' layout where the
elevation is stored in the third coordinate of a 3D linestring, and a
'line' layout where a separate feature is created in the target shapefile
for each segment of each feature in the source shapefile, with two
extra attributes compared to the original shapefile, one to store the
elevation, and another to store an arbitrary feature id referring to
the source feature in the source shapefile.

For the script to work, a configuration variable AHN_PATH must be set in
threedilib/localconfig.py pointing to the location of the elevation map,
and a variable INDEX_PATH pointing to the .shp file that contains the
index to the elevation map.
"""
import argparse
import logging
import math
import os
import sys

from scipy import ndimage
import numpy as np

from raster_tools import gdal
from raster_tools import groups
from raster_tools import ogr
from raster_tools import utils
from raster_tools import vectors

logger = logging.getLogger(__name__)

LAYOUT_POINT = 'point'
LAYOUT_LINE = 'line'

DRIVER = ogr.GetDriverByName(str('ESRI Shapefile'))
LINESTRINGS = ogr.wkbLineString, ogr.wkbLineString25D
MULTILINESTRINGS = ogr.wkbMultiLineString, ogr.wkbMultiLineString25D


def get_carpet(parameterized_line, distance, step):
    """
    Return M x N x 2 numpy array.

    It contains the first point of the first line, the centers, and
    the last point of the last line of the ParameterizedLine, but
    perpendicularly repeated by step along the normals to the segments
    of the ParameterizedLine, until distance is reached.
    """
    # length must be uneven, and no less than 2 * distance / step + 1
    steps = math.ceil(distance / step)
    offsets_1d = step * np.arange(-steps, steps + 1)

    # normalize and rotate the vectors of the linesegments
    rvectors = vectors.rotate(parameterized_line.vectors, 270)
    nvectors = vectors.normalize(rvectors)

    # extend vectors and centers
    evectors = np.concatenate([[nvectors[0]], nvectors[:], [nvectors[-1]]])
    ecenters = np.concatenate([[parameterized_line.points[0]],
                               parameterized_line.centers[:],
                               [parameterized_line.points[-1]]])

    offsets_2d = evectors.reshape(-1, 1, 2) * offsets_1d.reshape(1, -1, 1)
    points = offsets_2d + ecenters.reshape(-1, 1, 2)

    return points


def average_result(amount, lines, centers, values):
    """
    Return dictionary of numpy arrays.

    Points and values are averaged in groups of amount, but lines are
    converted per group to a line from the start point of the first line
    in the group to the end point of the last line in the group.
    """
    # determine the size needed to fit an integer multiple of amount
    oldsize = values.size
    newsize = int(np.ceil(values.size / amount) * amount)

    # determine lines
    ma_lines = np.ma.array(np.empty((newsize, 2, 2)), mask=True)
    ma_lines[:oldsize] = lines
    ma_lines[oldsize:] = lines[-1]  # Repeat last line
    result_lines = np.array([
        ma_lines.reshape(-1, amount, 2, 2)[:, 0, 0],
        ma_lines.reshape(-1, amount, 2, 2)[:, -1, 1],
    ]).transpose(1, 0, 2)

    # calculate points and values by averaging
    ma_centers = np.ma.array(np.empty((newsize, 2)), mask=True)
    ma_centers[:oldsize] = centers
    ma_values = np.ma.array(np.empty(newsize), mask=True)
    ma_values[:oldsize] = values
    return dict(lines=result_lines,
                values=ma_values.reshape(-1, amount).min(1),
                centers=ma_centers.reshape(-1, amount, 2).mean(1))


class Dataset(object):
    def __init__(self, dataset):
        """ Initialize from gdal dataset. """
        self.dataset = dataset

    def get_extent(self):
        """ Return tuple of xmin, ymin, xmax, ymax. """
        return (self.geotransform[0],
                self.geotransform[3] + self.size[1] * self.geotransform[5],
                self.geotransform[0] + self.size[0] * self.geotransform[1],
                self.geotransform[3])

    def get_cellsize(self):
        """ Return numpy array. """
        return np.array([[self.geotransform[1], self.geotransform[5]]])

    def get_origin(self):
        """ Return numpy array. """
        return np.array([[self.geotransform[0], self.geotransform[3]]])


class BaseProcessor(object):
    """ Base class for common processor methods. """
    def __init__(self, source, raster, target, **kwargs):
        self.source = source
        self.raster = raster
        self.target = target

        self.part = kwargs['part']
        self.width = kwargs['width']
        self.modify = kwargs['modify']
        self.average = kwargs['average']
        self.inverse = kwargs['inverse']
        self.distance = kwargs['distance']
        self.no_data_value = kwargs['no_data_value']
        self.elevation_attribute = kwargs['elevation_attribute']
        self.feature_id_attribute = kwargs['feature_id_attribute']

        if kwargs['modify'] and not kwargs['distance']:
            logger.warn('Warning: --modify used with zero distance.')

    def _modify(self, parameterized_line, points, values, step):
        """ Return dictionary of numpy arrays. """
        # first a minimum or maximum filter with requested width
        filtersize = round(self.width / step)
        if filtersize > 0:
            # choices based on inverse or not
            cval = values.max() if self.inverse else values.min()
            if self.inverse:
                extremum_filter = ndimage.maximum_filter
            else:
                extremum_filter = ndimage.minimum_filter
            # filtering
            fpoints = ndimage.convolve(
                points, np.ones((1, filtersize, 1)) / filtersize,
            )  # moving average for the points
            fvalues = extremum_filter(
                values, size=(1, filtersize), mode='constant', cval=cval,
            )  # moving extremum for the values
        else:
            fpoints = points
            fvalues = values

        if self.inverse:
            # find the minimum per filtered line
            index = (np.arange(len(fvalues)), fvalues.argmin(axis=1))
        else:
            # find the maximum per filtered line
            index = (np.arange(len(fvalues)), fvalues.argmax(axis=1))
        mpoints = fpoints[index]
        mvalues = fvalues[index]

        # sorting points and values according to projection on mline
        parameters = parameterized_line.project(mpoints)
        ordering = parameters.argsort()
        spoints = mpoints[ordering]
        svalues = mvalues[ordering]

        # quick 'n dirty way of getting to result dict
        rlines = np.array([spoints[:-1], spoints[1:]]).transpose(1, 0, 2)
        rcenters = spoints[1:]
        rvalues = svalues[1:]

        return {'lines': rlines, 'values': rvalues, 'centers': rcenters}

    def _calculate(self, wkb_line_string):
        """ Return lines, points, values tuple of numpy arrays. """
        # determine the point and values carpets
        geo_transform = self.raster.geo_transform

        # determine the points
        nodes = np.array(wkb_line_string.GetPoints())     # original nodes
        pline1 = vectors.ParameterizedLine(nodes[:, :2])  # parameterization
        pline2 = pline1.pixelize(geo_transform)           # add pixel edges

        # expand points when necessary
        if self.distance:
            step = geo_transform[1]
            points = get_carpet(step=step,
                                distance=self.distance,
                                parameterized_line=pline2)
        else:
            points = pline2.points.reshape(-1, 1, 2)

        # determine float indices
        x, y = points.transpose()
        p, a, b, q, c, d = geo_transform
        e, f, g, h = utils.get_inverse(a, b, c, d)

        # cast to integer indices
        j = np.uint32(e * (x - p) + f * (y - q))
        i = np.uint32(g * (x - p) + h * (y - q))

        # read corresponding values from raster
        bounds = (int(j.min()),
                  int(i.min()),
                  int(j.max()) + 1,
                  int(i.max()) + 1)
        array = self.raster.read(bounds)
        values = array[i - bounds[1], j - bounds[0]].transpose()

        # convert to desired no data values
        values[values == self.raster.no_data_value] = self.no_data_value

        # return lines, centers, values
        if self.modify:
            step = geo_transform[1]
            result = self._modify(step=step,
                                  points=points,
                                  values=values,
                                  parameterized_line=pline1)
        else:
            extremum = np.min if self.inverse else np.max
            result = {'lines': pline2.lines,
                      'centers': pline2.centers,
                      'values': extremum(values[1:-1], 1)}

        if self.average:
            return average_result(amount=self.average, **result)
        else:
            return result

    def _add_layer(self, layer):
        """ Add empty copy of layer. """
        srs = self.source.layer.GetSpatialRef()
        layer = self.target.CreateLayer(layer.GetName(), srs=srs)

        layer_defn = self.source.layer.GetLayerDefn()
        for i in range(layer_defn.GetFieldCount()):
            layer.CreateField(layer_defn.GetFieldDefn(i))
        self.layer = layer


class CoordinateProcessor(BaseProcessor):
    """ Writes a shapefile with height in z coordinate. """
    def _convert_wkb_line_string(self, source_wkb_line_string):
        """ Return a wkb line string. """
        result = self._calculate(wkb_line_string=source_wkb_line_string)
        target_wkb_line_string = ogr.Geometry(ogr.wkbLineString)

        # add the first point of the first line
        (x, y), z = result['lines'][0, 0], result['values'][0]
        target_wkb_line_string.AddPoint(float(x), float(y), float(z))

        # add the centers (x, y) and values (z)
        for (x, y), z in zip(result['centers'], result['values']):
            target_wkb_line_string.AddPoint(float(x), float(y), float(z))

        # add the last point of the last line
        (x, y), z = result['lines'][-1, 1], result['values'][-1]
        target_wkb_line_string.AddPoint(float(x), float(y), float(z))

        return target_wkb_line_string

    def _convert(self, source_geometry):
        """
        Return converted linestring or multiline.
        """
        geometry_type = source_geometry.GetGeometryType()
        if geometry_type in LINESTRINGS:
            return self._convert_wkb_line_string(source_geometry)
        if geometry_type in MULTILINESTRINGS:
            target_geometry = ogr.Geometry(source_geometry.GetGeometryType())
            for source_wkb_line_string in source_geometry:
                target_geometry.AddGeometry(
                    self._convert_wkb_line_string(source_wkb_line_string),
                )
            return target_geometry
        raise ValueError('Unexpected geometry type: {}'.format(
            source_geometry.GetGeometryName(),
        ))

    def _add_feature(self, feature):
        """ Add converted feature. """
        # create feature
        layer_definition = self.layer.GetLayerDefn()
        new_feature = ogr.Feature(layer_definition)

        # copy attributes
        for key, value in feature.items().items():
            new_feature[key] = value

        # set geometry and add to layer
        geometry = self._convert(source_geometry=feature.geometry())
        new_feature.SetGeometry(geometry)
        self.layer.CreateFeature(new_feature)

    def process(self):
        """ Convert dataset at path. """
        source, part = self.source, self.part
        self._add_layer(source.layer)
        for feature in source.select(part) if part else source:
            self._add_feature(feature)


class AttributeProcessor(BaseProcessor):
    """ Writes a shapefile with height in z attribute. """
    def _convert(self, source_geometry):
        """
        Return generator of (geometry, height) tuples.
        """
        geometry_type = source_geometry.GetGeometryType()
        if geometry_type in LINESTRINGS:
            source_wkb_line_strings = [source_geometry]
        elif geometry_type in MULTILINESTRINGS:
            source_wkb_line_strings = [line for line in source_geometry]
        else:
            raise ValueError('Unexpected geometry type: {}'.format(
                source_geometry.GetGeometryName(),
            ))
        for source_wkb_line_string in source_wkb_line_strings:
            result = self._calculate(wkb_line_string=source_wkb_line_string)
            for line, value in zip(result['lines'], result['values']):
                yield vectors.line2geometry(line), str(value)

    def _add_fields(self):
        """ Create extra fields. """
        for name, kind in ((str(self.elevation_attribute), ogr.OFTReal),
                           (str(self.feature_id_attribute), ogr.OFTInteger)):
            definition = ogr.FieldDefn(name, kind)
            self.layer.CreateField(definition)

    def _add_feature(self, feature_id, feature):
        """ Add converted features. """
        layer_definition = self.layer.GetLayerDefn()
        segments = self._convert(source_geometry=feature.geometry())
        for geometry, elevation in segments:
            # create feature

            new_feature = ogr.Feature(layer_definition)

            # copy attributes
            for key, value in feature.items().items():
                new_feature[key] = value

            # add special attributes
            new_feature[str(self.elevation_attribute)] = elevation
            new_feature[str(self.feature_id_attribute)] = feature_id

            # set geometry and add to layer
            new_feature.SetGeometry(geometry)
            self.layer.CreateFeature(new_feature)

    def process(self):
        """ Convert dataset at path. """
        source, part = self.source, self.part
        self._add_layer(self.source.layer)
        self._add_fields()

        selection = source.select(part) if part else source
        for feature_id, feature in enumerate(selection):
            self._add_feature(feature_id=feature_id, feature=feature)


def process(raster, source, target, **kwargs):
    if kwargs.pop('layout') == LAYOUT_POINT:
        Processor = CoordinateProcessor
    else:
        Processor = AttributeProcessor
    Processor(raster=raster, source=source, target=target, **kwargs).process()


def line_up(source_path, raster_path, target_path, **kwargs):
    """
    Take linestrings from source and create target with height added.

    Source and target are both shapefiles.
    """
    # target
    if os.path.exists(target_path):
        if kwargs.pop('overwrite'):
            DRIVER.DeleteDataSource(str(target_path))
        else:
            logger.info('"%s" already exists. Use --overwrite.', target_path)
            return
    target = DRIVER.CreateDataSource(str(target_path))

    # rasters
    if os.path.isdir(raster_path):
        datasets = [gdal.Open(os.path.join(raster_path, path))
                    for path in sorted(os.listdir(raster_path))]
    else:
        datasets = [gdal.Open(raster_path)]
    raster = groups.Group(*datasets)

    # source
    source = utils.PartialDataSource(source_path)

    process(raster=raster, source=source, target=target, **kwargs)


def get_parser():
    """ Return arguments dictionary. """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('raster_path',
                        metavar='RASTER',
                        help='Path to gdal dataset.')
    parser.add_argument('source_path',
                        metavar='SOURCE',
                        help='Path to shapefile with 2D linestrings.')
    parser.add_argument('target_path',
                        metavar='TARGET',
                        help='Path to target shapefile.')
    parser.add_argument('-o', '--overwrite',
                        action='store_true',
                        help='Overwrite TARGET if it exists.')
    parser.add_argument('-d', '--distance',
                        metavar='DISTANCE',
                        type=float,
                        default=0,
                        help=('Distance (half-width) to look '
                              'perpendicular to the segments to '
                              'find the highest (or lowest, with '
                              ' --inverse) points on the elevation '
                              ' map. Defaults to 0.0.'))
    parser.add_argument('-w', '--width',
                        metavar='WIDTH',
                        type=float,
                        default=0,
                        help=('Guaranteed width of maximum. '
                              'Defaults to 0.0.'))
    parser.add_argument('-m', '--modify',
                        action='store_true',
                        help='Change horizontal geometry.')
    parser.add_argument('-a', '--average',
                        metavar='AMOUNT',
                        type=int,
                        default=0,
                        help='Average of points and minimum (or'
                             ' maximum, with --inverse) of values.')
    parser.add_argument('-l', '--layout',
                        metavar='LAYOUT',
                        choices=[LAYOUT_POINT, LAYOUT_LINE],
                        default=LAYOUT_POINT,
                        help="Target shapefile layout.")
    parser.add_argument('-f', '--feature-id-attribute',
                        metavar='FEATURE_ID_ATTRIBUTE',
                        default='_feat_id',
                        help='Attribute name for the feature id.')
    parser.add_argument('-e', '--elevation-attribute',
                        metavar='ELEVATION_ATTRIBUTE',
                        default='_elevation',
                        help='Attribute name for the elevation.')
    parser.add_argument('-i', '--inverse',
                        action='store_true',
                        help='Look for lowest points instead of highest.')
    parser.add_argument('-p', '--part',
                        help='partial processing source, for example "2/3"')
    parser.add_argument('-n', '--no-data-value',
                        type=float,
                        default=-9999.,
                        help='no data value for no data pixels.')
    return parser


def main():
    """ Call line_up with commandline args. """
    logging.basicConfig(stream=sys.stderr,
                        level=logging.DEBUG,
                        format='%(message)s')
    line_up(**vars(get_parser().parse_args()))


if __name__ == '__main__':
    exit(main())
