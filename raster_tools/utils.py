# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import urllib

from osgeo import ogr

logger = logging.getLogger(__name__)


def get_geo_transforms(index_path):
    """
    Return dictionary mapping leaf number to geotransform.
    """
    def get_geo_transform(feature):
        """ Return a feature's geo_transform. """
        x1, x2, y1, y2 = feature.geometry().GetEnvelope()
        return x1, 0.5, 0.0, y2, 0.0, -0.5

    ogr_index_datasource = ogr.Open(index_path)
    ogr_index_layer = ogr_index_datasource[0]

    return {ogr_index_feature[b'BLADNR'][1:]:
            get_geo_transform(ogr_index_feature)
            for ogr_index_feature in ogr_index_layer}
