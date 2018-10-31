raster-tools
============

A collection of raster tools.


Local setup
-----------

The docker-compose file expects the following folders to exist:

 - ``~/.cache/pip``
 - ``~/.cache/pipenv``

Check if they exist and if you are the owner. If they do not exist, create them
with ``mkdir``, and if they are not owned by you, use ``sudo chown``.


Local development
-----------------

First, clone this repo and make some required directories::

    $ git clone git@github.com:nens/raster-tools
    $ cd raster-tools

Create a docker-compose.override.yaml to map local filesystems:

.. code-block:: yaml

    version: '3'
    services:
      lib:
        volumes:
          - /some/local/path:/some/container/path

Then build the docker image, providing your user and group ids for correct file
permissions::

    $ docker-compose build --build-arg uid=`id -u` --build-arg gid=`id -g` lib

The entrypoint into the docker is set to `pipenv run`, so that every command is
executed in the pipenv-managed virtual environment. On the first
`docker-compose run`, the `.venv` folder will be created automatically inside
your project directory::

    $ docker-compose run --rm lib bash

Then install the packages (including dev packages) listed in `Pipfile.lock`::

    (docker) $ pipenv sync --dev

Now you are ready to run the raster tools in the container.


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


Filling nodata in rasters
-------------------------

Use fillnodata to fill nodata regions in rasters. The input to the algorithm is
the edge surrounding the region. The raster saved as the target argument only
contains the complementary cells.


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
