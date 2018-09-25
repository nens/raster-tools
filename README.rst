raster-tools
==========================================

Installation
------------
1. Install dependencies: ``sudo apt install python-dev python-pip libgdal-dev libpq-dev``
2. Upgrade pip: ``sudo pip install setuptools pip --upgrade``
3. Install pipenv: ``sudo pip install pipenv``
4. ``cd`` to the raster-tools directory
5. Run ``PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy``
6. Add the ``raster-tools/.venv/bin`` directory to PATH in your ``/etc/environment``.


Rasterizing landuse tables
--------------------------
For rasterization of landuse tables from a postgres datasource a special
wrapper command is available at bin/rasterize-landuse, use --help for args.


Creating a seamless large-scale void-filled raster (deprecated)
---------------------------------------------------------------
1. Make index with power-of-two dimensions for some extent [reindex] 
2. Aggregate base dataset to that index for 32x32 pixels [aggregate]
3. Merge result of 2. into single raster [gdal_merge.py]
4. Fill result of 3. with fillnodata algorithm [fillnodata]
5. Combine with result of 3. to single filled dem [gdalwarp]
6. Fill base data using result of 5. as
   'ceiling' and result of 1. as index [mfillnodata]
7. Cut result back into desired tiling using [retile]

Procedure for filling completely filling internal nodata
--------------------------------------------------------
1. Make VRT of all source rasters
2. Create filled rasters using fillnodata using source (1st pass)
    - use units shape for partitioning
    - keep edge geometries (an attribute for unit names?)
3. Dissolve edge geometries and give them proper names
4. Create filled rasters for the dissolved edge geometries (2nd pass)
5. Have a separate script spatially lookup all 1st pass outputs and put
   2nd pass outputs into it

Creating streamlines
--------------------

flow-fil index raster cover output/f                        # fill depressions
flow-dir index output/f/all.vrt cover output/d              # derive direction
flow-acc index output/d/all.vrt output/a                    # accumulate
flow-vec index output/d/all.vrt output/a/all.vrt output/v   # makes features
flow-rst index output/v output/r                            # features to tifs


Post-nensskel setup TODO
------------------------

Here are some instructions on what to do after you've created the project with
nensskel.

- Add a new jenkins job at
  http://buildbot.lizardsystem.nl/jenkins/view/djangoapps/newJob or
  http://buildbot.lizardsystem.nl/jenkins/view/libraries/newJob . Job name
  should be "raster-tools", make the project a copy of the existing "lizard-wms"
  project (for django apps) or "nensskel" (for libraries). On the next page,
  change the "github project" to ``https://github.com/nens/raster-tools/`` and
  "repository url" fields to ``git@github.com:nens/raster-tools.git`` (you might
  need to replace "nens" with "lizardsystem"). The rest of the settings should
  be OK.

- The project is prepared to be translated with Lizard's
  `Transifex <http://translations.lizard.net/>`_ server. For details about
  pushing translation files to and fetching translation files from the
  Transifex server, see the ``nens/translations`` `documentation
  <https://github.com/nens/translations/blob/master/README.rst>`_.
