# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-

from osgeo import ogr
import numpy as np


def point2geometry(point):
    """ Return geometry. """
    geometry = ogr.Geometry(ogr.wkbPoint)
    geometry.AddPoint_2D(*map(float, point))
    return geometry


def line2geometry(line):
    """ Return geometry. """
    geometry = ogr.Geometry(ogr.wkbLineString)
    for point in line:
        geometry.AddPoint_2D(*map(float, point))
    return geometry


def polygon2geometry(line):
    """ Return geometry. """
    geometry = ogr.Geometry(ogr.wkbPolygon)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for point in line:
        ring.AddPoint_2D(*map(float, point))
    geometry.AddGeometry(ring)
    return geometry


def magnitude(vectors):
    """ Return magnitudes. """
    return np.sqrt((vectors ** 2).sum(1))


def normalize(vectors):
    """ Return unit vectors. """
    return vectors / magnitude(vectors).reshape(-1, 1)


def rotate(vectors, degrees):
    """ Return vectors rotated by degrees. """
    return np.vstack([
        +np.cos(np.radians(degrees)) * vectors[:, 0] +
        -np.sin(np.radians(degrees)) * vectors[:, 1],
        +np.sin(np.radians(degrees)) * vectors[:, 0] +
        +np.cos(np.radians(degrees)) * vectors[:, 1],
    ]).transpose()


class ParameterizedLine(object):
    """
    LineString with handy parameterization and projection properties.
    """
    def __init__(self, points):
        # data
        self.points = np.array(points)

        # views
        self.p = self.points[:-1]
        self.q = self.points[1:]
        self.lines = np.hstack([self.p, self.q]).reshape(-1, 2, 2)

        # derived
        self.length = len(points) - 1
        self.vectors = self.q - self.p
        self.centers = (self.p + self.q) / 2

    def __getitem__(self, parameters):
        """ Return points corresponding to parameters. """
        i = np.uint64(np.where(parameters == self.length,
                               self.length - 1, parameters))
        t = np.where(parameters == self.length,
                     1, np.remainder(parameters, 1)).reshape(-1, 1)
        return self.p[i] + t * self.vectors[i]

    def pixelize(self, geo_transform):
        """
        Return ParameterizedLine instance with points added at any
        boundary crossing of the raster defined by geo_transform.
        """
        p, a, b, q, c, d = geo_transform
        if p % a or q % d or b or c or a + d:
            raise ValueError('Currently only aligned, '
                             'square pixels are implemented')
        size = a

        extent = np.array([self.points.min(0), self.points.max(0)])
        parameters = []
        # loop dimensions for intersection parameters
        for i in 0, 1:
            intersects = np.arange(
                size * np.ceil(extent[0, i] / size),
                size * np.ceil(extent[1, i] / size),
                size,
            ).reshape(-1, 1)
            # calculate intersection parameters for each vector
            nonzero = self.vectors[:, i].nonzero()
            lparameters = ((intersects - self.p[nonzero, i])
                           / self.vectors[nonzero, i])
            # add integer to parameter and mask outside line
            global_parameters = np.ma.array(
                np.ma.array(lparameters + np.arange(nonzero[0].size)),
                mask=np.logical_or(lparameters < 0, lparameters > 1),
            )
            # only unmasked values must be in parameters
            parameters.append(global_parameters.compressed())

        # add parameters for original points
        parameters.append(np.arange(self.length + 1))

        # apply unique on single precision, eliminating really close points
        unique = np.unique(np.concatenate(parameters).astype('f4'))

        return ParameterizedLine(self[unique])

    def project(self, points):
        """
        Return array of parameters.

        Find closest projection of each point on the magic line.
        """
        # some reshapings
        a = self.p.reshape(1, -1, 2)
        b = self.q.reshape(1, -1, 2)
        c = points.reshape(-1, 1, 2)

        # some vectors
        vab = b - a
        vac = c - a
        vabn = normalize(vab[0]).reshape(1, -1, 2)

        # perform dot product and calculations
        dotprod = np.sum(vac * vabn, axis=2).reshape(len(points), -1, 1)
        vabl = magnitude(vab[0]).reshape(1, -1, 1)
        lparameters = (dotprod / vabl)[..., 0].round(3)  # what round to take?

        # add integer to parameter and mask outside line
        gparameters = np.ma.array(
            np.array(lparameters
                     + np.arange(len(self.vectors)).reshape(1, -1)),
            mask=np.logical_or(lparameters < 0, lparameters > 1),
        )

        # calculate distances and sort accordingly
        projections = dotprod * vabn + a
        distances = np.ma.array(
            magnitude((c - projections).reshape(-1, 2)),
            mask=gparameters.mask,
        ).reshape(len(points), -1)
        closest = gparameters[(np.arange(len(distances)), distances.argmin(1))]

        if closest.mask.any():
            raise ValueError('Masked values in projection.')

        return closest.data


def array2polygon(array):
    """
    Return a polygon geometry.

    This method numpy to prepare a wkb string. Seems only faster for
    larger polygons, compared to adding points individually.
    """
    # 13 bytes for the header, 16 bytes per point
    nbytes = 13 + 16 * array.shape[0]
    data = np.empty(nbytes, dtype=np.uint8)
    # little endian
    data[0:1] = 1
    # wkb type, number of rings, number of points
    data[1:13].view('u4')[:] = (3, 1, array.shape[0])
    # set the points
    data[13:].view('f8')[:] = array.ravel()
    return ogr.CreateGeometryFromWkb(data.tostring())


def array2multipoint(array):
    """
    Return a 3d multipoint geometry.

    This method numpy to prepare a wkb string. Performance not tested.
    """
    npoints = len(array)
    head = np.empty(9, dtype='u1')
    head[0] = 0                                             # endianness
    head[1:5] = 128, 0, 0, 4                                # wkb multipoint
    head[5:9].view('i4')[:] = np.int32(npoints).byteswap()  # amount of points

    bulk = np.empty((npoints, 29), 'u1')
    bulk[:, 0] = 0                                          # unknown
    bulk[:, 1:5] = 128, 0, 0, 1                             # wkb point
    bulk[:, 5:] = array.astype('f8').view('i8').byteswap().view('u1')
    return ogr.CreateGeometryFromWkb(head.tostring() + bulk.tostring())
