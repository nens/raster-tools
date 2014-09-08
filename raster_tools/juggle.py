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
import Queue
import shlex
import signal
import subprocess
import sys

REPORT = 'executions: {}, processed: {}, existed: {}.'

logger = logging.getLogger(__name__)
queue = Queue.Queue(maxsize=1)


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
    """
    Execute the requested command.
    """
    queue.put(subprocess.Popen(shlex.split(command), preexec_fn=pre).wait())


def callback(result):
    """
    Put None on the queue, to signal the end of the juggling.
    """
    queue.put(None)


def command(script, target):
    """
    Parallel processing of a list of sources.
    """
    processes = multiprocessing.cpu_count()
    template = '{} {{}} {}'.format(' '.join(script), target)

    # execute all using a pool
    commands = (template.format(source) for source in sys.stdin)
    pool = ThreadPool(processes=processes)
    pool.map_async(execute, commands, callback=callback)

    outcome = {0: 0, 1: 0}
    for executions in itertools.count(1):
        # take from queue until None comes out
        q = queue.get()
        if q is None:
            break

        # log report for batch
        outcome[q] += 1
        logger.debug(REPORT.format(executions, outcome[0], outcome[1]))


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    return command(**vars(get_parser().parse_args()))
