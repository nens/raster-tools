raster-tools
============

Development installation
------------------------

Create a docker-compose.override.yaml to add local filesystems:

.. code-block:: yaml

    version: '3'
    services:
      lib:
        volumes:
          - /some/local/path:/some/container/path


On your machine::

    $ docker-compose up --no-start
    $ docker-compose start
    $ docker-compose exec lib bash

And then in the container::

    $ pipenv sync --dev
    $ pipenv shell

In this pipenv shell the raster-tools commands can be run in the container.


Task server installation
------------------------

1. Install dependencies::

    $ sudo apt install\
        python3-dev\
        python-pip\
        libgdal-dev\
        libpq-dev\

2. Upgrade python packages::

    $ sudo pip install --upgrade pip pipenv setuptools

3. Clone this repository and step into it.

3. Run ``PIPENV_VENV_IN_PROJECT=1 pipenv sync --dev``

4. Add the absolute path to ``.venv/bin`` to the PATH in ``/etc/environment``.


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
