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

TODO - no more index needed, instead, single raster - support for nodata,
optional fillvalue """

import argparse
import os

from raster_tools import gdal
from raster_tools import ogr
from scipy import ndimage
import numpy as np

from raster_tools import vectors

LAYOUT_POINT = 'point'
LAYOUT_LINE = 'line'

PIXELSIZE = 0.5  # AHN2
STEPSIZE = 0.5  # for looking perpendicular to line.

LINESTRINGS = ogr.wkbLineString, ogr.wkbLineString25D
MULTILINESTRINGS = ogr.wkbMultiLineString, ogr.wkbMultiLineString25D


def get_carpet(mline, distance, step=None):
    """
    Return MxNx2 numpy array.

    It contains the first point of the first line, the centers, and the
    last point of the last line of the MagicLine, but perpendicularly
    repeated along the normals to the segments of the MagicLine, up to
    distance, with step.
    """
    # determine the offsets from the points on the line
    if step is None or step == 0:
        length = 2
    else:
        # length must be uneven, and no less than 2 * distance / step + 1
        # take a look at np.around() (round to even values)!
        length = 2 * np.round(0.5 + distance / step) + 1
    offsets_1d = np.mgrid[-distance:distance:length * 1j]
    # normalize and rotate the vectors of the linesegments
    nvectors = vectors.normalize(vectors.rotate(mline.vectors, 270))

    # extend vectors and centers
    evectors = np.concatenate([[nvectors[0]], nvectors[:], [nvectors[-1]]])
    ecenters = np.concatenate([[mline.points[0]],
                               mline.centers[:],
                               [mline.points[-1]]])

    offsets_2d = evectors.reshape(-1, 1, 2) * offsets_1d.reshape(1, -1, 1)
    points = offsets_2d + ecenters.reshape(-1, 1, 2)

    return points


def paste_values(points, values, leafno):
    """ Paste values of evelation pixels at points. """
    dataset = get_dataset(leafno)
    xmin, ymin, xmax, ymax = dataset.get_extent()
    cellsize = dataset.get_cellsize()
    origin = dataset.get_origin()

    # determine which points are outside leaf's extent
    # '=' added for the corner where the index origin is
    index = np.logical_and(np.logical_and(points[..., 0] >= xmin,
                                          points[..., 0] < xmax),
                           np.logical_and(points[..., 1] > ymin,
                                          points[..., 1] <= ymax))
    # determine indices for these points
    indices = tuple(np.uint64(
        (points[index] - origin) / cellsize,
    ).transpose())[::-1]

    # assign data for these points to corresponding values
    values[index] = dataset.data[indices]


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


class BaseWriter(object):
    """ Base class for common writer methods. """
    def __init__(self, target_path, raster_path, **kwargs):
        self.dataset = gdal.Open(raster_path)
        self.path = target_path
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __enter__(self):
        """ Creates or replaces the target shapefile. """
        driver = ogr.GetDriverByName(b'ESRI Shapefile')
        if os.path.exists(self.path):
            driver.DeleteDataSource(str(self.path))
        self.dataset = driver.CreateDataSource(str(self.path))
        return self

    def __exit__(self, type, value, traceback):
        """ Close dataset. """
        self.layer = None
        self.dataset = None

    def _modify(self, points, values, mline, step):
        """ Return dictionary of numpy arrays. """
        # First a minimum or maximum filter with requested width
        filtersize = np.round(self.width / step)
        if filtersize > 0:
            # Choices based on inverse or not
            cval = values.max() if self.inverse else values.min()
            if self.inverse:
                extremum_filter = ndimage.maximum_filter
            else:
                extremum_filter = ndimage.minimum_filter
            # Filtering
            fpoints = ndimage.convolve(
                points, np.ones((1, filtersize, 1)) / filtersize,
            )  # Moving average for the points
            fvalues = extremum_filter(
                values, size=(1, filtersize), mode='constant', cval=cval,
            )  # Moving extremum for the values
        else:
            fpoints = points
            fvalues = values

        if self.inverse:
            # Find the minimum per filtered line
            index = (np.arange(len(fvalues)), fvalues.argmin(axis=1))
        else:
            # Find the maximum per filtered line
            index = (np.arange(len(fvalues)), fvalues.argmax(axis=1))
        mpoints = fpoints[index]
        mvalues = fvalues[index]

        # Sorting points and values according to projection on mline
        parameters = mline.project(mpoints)
        ordering = parameters.argsort()
        spoints = mpoints[ordering]
        svalues = mvalues[ordering]

        # Quick 'n dirty way of getting to result dict
        rlines = np.array([spoints[:-1], spoints[1:]]).transpose(1, 0, 2)
        rcenters = spoints[1:]
        rvalues = svalues[1:]

        return dict(lines=rlines,
                    centers=rcenters,
                    values=rvalues)

    def _calculate(self, wkb_line_string):
        """ Return lines, points, values tuple of numpy arrays. """
        # Determine the leafnos
        mline = vectors.MagicLine(np.array(wkb_line_string.GetPoints())[:, :2])
        leafnos = get_leafnos(mline=mline, distance=self.distance)

        # Determine the point and values carpets
        pline = mline.pixelize(size=PIXELSIZE)
        points = get_carpet(
            mline=pline,
            distance=self.distance,
            step=STEPSIZE,
        )
        values = np.ma.array(
            np.empty(points.shape[:2]),
            mask=True,
        )

        # Get the values into the carpet per leafno
        # TODO put values from raster in carpet
        for leafno in leafnos:
            paste_values(points, values, leafno)

        if values.mask.any():
            raise ValueError('Masked values remaining after filling!')

        # Return lines, centers, values
        if self.modify:
            result = self._modify(points=points,
                                  values=values,
                                  mline=mline,
                                  step=STEPSIZE)
        else:
            result = dict(lines=pline.lines,
                          centers=pline.centers,
                          values=values.data[1:-1].max(1))

        if self.average:
            return average_result(amount=self.average, **result)
        else:
            return result

    def _add_layer(self, layer):
        """ Add empty copy of layer. """
        # Create layer
        self.layer = self.dataset.CreateLayer(layer.GetName())
        # Copy field definitions
        layer_definition = layer.GetLayerDefn()
        for i in range(layer_definition.GetFieldCount()):
            self.layer.CreateField(layer_definition.GetFieldDefn(i))


class CoordinateWriter(BaseWriter):
    """ Writes a shapefile with height in z coordinate. """
    def _convert_wkb_line_string(self, source_wkb_line_string):
        """ Return a wkb line string. """
        result = self._calculate(wkb_line_string=source_wkb_line_string)
        target_wkb_line_string = ogr.Geometry(ogr.wkbLineString)

        # Add the first point of the first line
        (x, y), z = result['lines'][0, 0], result['values'][0]
        target_wkb_line_string.AddPoint(float(x), float(y), float(z))

        # Add the centers (x, y) and values (z)
        for (x, y), z in zip(result['centers'], result['values']):
            target_wkb_line_string.AddPoint(float(x), float(y), float(z))

        # Add the last point of the last line
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
        # Create feature
        layer_definition = self.layer.GetLayerDefn()
        new_feature = ogr.Feature(layer_definition)
        # Copy attributes
        for key, value in feature.items().items():
            new_feature[key] = value
        # Set geometry and add to layer
        geometry = self._convert(source_geometry=feature.geometry())
        new_feature.SetGeometry(geometry)
        self.layer.CreateFeature(new_feature)
        self.indicator.update()

    def add(self, path, **kwargs):
        """ Convert dataset at path. """
        dataset = ogr.Open(path)
        count = sum(layer.GetFeatureCount() for layer in dataset)
        self.indicator = progress.Indicator(count)
        for layer in dataset:
            self._add_layer(layer)
            for feature in layer:
                try:
                    self._add_feature(feature)
                except Exception as e:
                    with open('errors.txt', 'a') as errorfile:
                        errorfile.write(unicode(e) + '\n')
                        errorfile.write(unicode(feature.items()) + '\n')
        dataset = None


class AttributeWriter(BaseWriter):
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
        generator = self._convert(source_geometry=feature.geometry())
        for geometry, elevation in generator:
            # Create feature
            new_feature = ogr.Feature(layer_definition)
            # Copy attributes
            for key, value in feature.items().items():
                new_feature[key] = value
            # Add special attributes
            new_feature[str(self.elevation_attribute)] = elevation
            new_feature[str(self.feature_id_attribute)] = feature_id
            # Set geometry and add to layer
            new_feature.SetGeometry(geometry)
            self.layer.CreateFeature(new_feature)
        self.indicator.update()

    def add(self, path):
        """ Convert dataset at path. """
        dataset = ogr.Open(path)
        count = sum(layer.GetFeatureCount() for layer in dataset)
        self.indicator = progress.Indicator(count)
        for layer in dataset:
            self._add_layer(layer)
            self._add_fields()
            for feature_id, feature in enumerate(layer):
                try:
                    self._add_feature(feature_id=feature_id, feature=feature)
                except Exception as e:
                    with open('errors.txt', 'a') as errorfile:
                        errorfile.write(unicode(e) + '\n')
                        errorfile.write(unicode(feature.items()) + '\n')
        dataset = None


def addheight(raster_path, source_path, target_path, overwrite,
              distance, width, modify, average, inverse,
              layout, elevation_attribute, feature_id_attribute):
    """
    Take linestrings from source and create target with height added.

    Source and target are both shapefiles.
    """
    if os.path.exists(target_path) and not overwrite:
        print("'{}' already exists. Use --overwrite.".format(target_path))
        return 1

    if modify and not distance:
        print('Warning: --modify used with zero distance.')

    Writer = CoordinateWriter if layout == LAYOUT_POINT else AttributeWriter
    with Writer(raster_path,
                target_path,
                distance=distance,
                width=width,
                modify=modify,
                average=average,
                inverse=inverse,
                elevation_attribute=elevation_attribute,
                feature_id_attribute=feature_id_attribute) as writer:
        writer.add(source_path)
    return 0


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
                        help='Average of points and minimum of values.')
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
    return parser


def main():
    """ Call addheight() with commandline args. """
    addheight(**vars(get_parser().parse_args()))


cache = {}  # Contains leafno's and the index

if __name__ == '__main__':
    exit(main())
