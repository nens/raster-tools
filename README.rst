raster-tools
==========================================

Installation
------------
1. Install dependencies::

    sudo apt install\
        python-dev\
        python-pip\
        python-tk\
        libgdal-dev\
        libpq-dev\

2. Upgrade pip: ``sudo pip install setuptools pip --upgrade``
3. Install pipenv: ``sudo pip install pipenv``
4. ``cd`` to the raster-tools directory
5. Run ``PIPENV_VENV_IN_PROJECT=1 pipenv install --deploy``
6. Add the ``raster-tools/.venv/bin`` directory to PATH in your ``/etc/environment``.


Rasterizing landuse tables
--------------------------
For rasterization of landuse tables from a postgres datasource a special
wrapper command is available at bin/rasterize-landuse, use --help for args.


Creating streamlines
--------------------

Run the following scripts for streamline calculation::

    flow-fil  # depression filling
    flow-dir  # direction calculation
    flow-acc  # accumulation
    flow-vec  # make shapefiles
    flow-rst  # make rasters from shapefiles
