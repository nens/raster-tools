# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from datetime import datetime as Datetime
from os import environ
from os.path import basename
from sys import argv

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

# learn which scripts are in use and should be ported to py3
user = environ.get("USER", "anonymous")
script = basename(argv[0])
time = Datetime.now().strftime("%Y:%m:%d %H:%M:%S")
logpath = "/var/tmp/raster-tools-py2.log"
record = ",".join([time, user, script])
with open(logpath, "a") as f:
    f.write(record + "\n")
