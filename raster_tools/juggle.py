# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from multiprocessing.pool import ThreadPool

import argparse
import itertools
import logging
import multiprocessing
import shlex
import signal
import subprocess
import sys

REPORT = 'executions: {}, processed: {}, existed: {}.'
logger = logging.getLogger(__name__)


def get_parser():
    """ Return arguments dictionary. """
    parser = argparse.ArgumentParser(description='No description yet.')
    parser.add_argument('target')
    parser.add_argument('script', nargs='*')
    return parser


def pre():
    """ Execute this as preexec in subprocess, to prevent interruption. """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def execute(command):
    """ Execute the requested command. """
    return subprocess.Popen(shlex.split(command), preexec_fn=pre).wait()


def command(script, target):
    """ Main command. """
    command = '{} {{}} {}'.format(' '.join(script), target)
    # Going to supply it in batches to the thread pool.
    processes = multiprocessing.cpu_count()
    pool = ThreadPool(processes=processes)

    executions, existed, processed = 0, 0, 0
    sources = list(itertools.islice(sys.stdin, 0, processes))
    while sources:
        # generate and execute
        commands = (command.format(source) for source in sources)
        codes = pool.map(execute, commands)
        logger.debug('batch done')

        # log report for batch
        executions += len(codes)
        existed += codes.count(1)
        processed += codes.count(0)
        logger.debug(REPORT.format(executions, processed, existed))

        sources = list(itertools.islice(sys.stdin, 0, processes))


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return command(**vars(get_parser().parse_args()))
