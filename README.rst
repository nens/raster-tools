raste-tools
============

A collection of raster tools.


Docker installation
------------------------

For development, you can use a docker-compose setup::

    $ docker compose build --build-arg uid=`id -u` --build-arg gid=`id -g`
    $ docker compose up --no-start
    $ docker compose start
    $ docker compose exec lib pipx ensurepath
    $ docker compose exec lib pipx install --system-site-packages --editable .
    $ docker compose exec lib bash


Note that the /mnt folder is already mapped to the host mount.


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

Streamlines have been rasterized in the past to be able to visualize them with
good performance on any zoomlevel. A number of tricks can be to make them look
like a vector dataset:

- Rasterize at sufficiently high resolution so that supersampling is never
  needed
- (Pre)aggregate using a `maximum` algorithm for lower resolutions
- Use binary dilation after serving image tiles on one or more higher classes
  to create a 'wider stroke' effect


Multiprocessing
---------------

A number of scripts have a ``--part`` option to run the script on a subset of the
source features, e.g. --part=2/3 to run on the second part of three parts. To
use xargs to run some script for example on 4 processes, use::

    xargs -a <(echo -e '1/4\n2/4\n3/4\n4/4') -L 1 -P 4 your_script --your_args --part
