raste-tools
============

A collection of raster tools.


Development installation
------------------------

For development, you can use a docker-compose setup::

    $ docker-compose build --build-arg uid=`id -u` --build-arg gid=`id -g` lib
    $ docker-compose up --no-start
    $ docker-compose start
    $ docker-compose exec lib bash

Create a virtualenv, install dependencies & package, run tests::

    # note that Dockerfile prepends .venv/bin to $PATH
    (docker)$ virtualenv --system-site-packages .venv 
    (docker)$ pip install -r requirements.txt --index-url https://packages.lizard.net
    (docker)$ pip install -e .[test]
    (docker)$ pytest

Update packages::
    
    (docker)$ rm -rf .venv
    (docker)$ virtualenv --system-site-packages .venv
    (docker)$ pip install -e . --index-url https://packages.lizard.net
    (docker)$ pip freeze | grep -v raster-tools > requirements.txt

Now you are ready to run the raster tools in the container.


Mapping an extra local folder
-----------------------------

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


Task server installation
------------------------

One time server provisioning::

    ansible-playbook ansible/provision.yml -i ansible/task.yml

Deploying new versions::

    ansible-playbook ansible/deploy.yml -i ansible/task.yml 

Be aware that if you provision a server for the first time, the authentication
needs to be setup. We may need a credentials file to access private packages on
packages.lizard.net, to be put in `deploy/files/nens_netrc` file.

To enable users to use the raster-tools scripts, append the absolute path to
``.venv/bin`` to the PATH in ``/etc/environment``.


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
