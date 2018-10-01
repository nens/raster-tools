raster-tools
============

Installation
------------

1. Install dependencies::

    $ sudo apt install\
        python-dev\
        python-pip\
        libgdal-dev\
        libpq-dev\

2. Upgrade python packages::

    $ sudo pip install --upgrade pip pipenv setuptools

3. ``cd`` to the raster-tools directory
4. Depending on your ubuntu version, make symlinks::

    $ ln -s Pipfile.1604 Pipfile
    $ ln -s Pipfile.lock.1604 Pipfile.lock

   or::

    $ ln -s Pipfile.1804 Pipfile
    $ ln -s Pipfile.lock.1804 Pipfile.lock


5. Run ``PIPENV_VENV_IN_PROJECT=1 pipenv sync --dev``
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
