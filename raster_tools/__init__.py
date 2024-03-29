# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.

from datetime import datetime as Datetime
from os import chmod, environ
from os.path import basename, exists
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
time = Datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logpath = "/var/tmp/raster-tools.log"
record = ",".join([time, user, script])

# create and set permissions if needed
if not exists(logpath):
    with open(logpath, "w") as f:
        pass
    chmod(logpath, 0o666)

# append the record
with open(logpath, "a") as f:
    f.write(record + "\n")
