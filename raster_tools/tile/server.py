# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans, see LICENSE.rst.
""" TODO Docstring. """

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import flask

from raster_tools.tile import storages

logger = logging.getLogger(__name__)
app = flask.Flask(__name__)


@app.route('/<path:path>/<int:z>/<int:x>/<int:y>')
def tile(path, x, y, z):
    """ Return image. """
    # read
    storage = storages.ZipFileStorage(path=path)
    try:
        content = storage[x, y, z]
    except KeyError:
        flask.abort(404)

    # mime
    if content[1:4] == str('PNG'):
        content_type = 'image/png'
    else:
        content_type = 'image/jpeg'

    return content, 200, {'content-type': content_type,
                          'Access-Control-Allow-Origin': '*',
                          'Access-Control-Allow-Methods': 'GET'}


def server():
    """ Run a development server with console logging. """
    app.run(host='0.0.0.0', debug=True)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


def main():
    """ Call server with args from parser. """
    # logging
    kwargs = vars(get_parser().parse_args())
    if kwargs.pop('verbose'):
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level, format='%(message)s')

    # run or fail
    try:
        server(**kwargs)
        return 0
    except SystemExit:
        raise  # that's how flasks reloader works
    except:
        logger.exception('An exception has occurred.')
        return 1
