# -*- coding: utf-8 -*-
"""
Compute a shapefile that distinguishes "plas", "overlast", and "modelfout" from
two 3Di gridadmin and results files ("piek" and "blok").

This analysis has 4 steps:

1. Remove connections between nodes that have a flow area
   ("doorstroomoppervlak") below the parameter min_flow_area.
2. Split the remaining connections in two groups based on their gradient
   ("verhang"), according to the max_gradient parameter.
3. Arrange 2D nodes into groups that are connected with connections that have
   a low gradient ("verhang") according to step 2.
4. Classify each group into 3 categories:
   - "overlast" if they have a low-gradient connection to a 1D node
   - "modelfout" if they have only high-gradient connections to a 1D node
   - "plas" if they have no connections to a 1D node

The analysis is done on the last timeframe of two 3Di result files ("piek" en
"blok"). The resulting shapefile consists of polygons of 2D nodes with the
fields "case_blok", "case_piek" and "case_final" added.

"case_final" is determined from "case_block" and "case_piek", as follows:
 - "overlast" if "case_piek" OR "case_blok" is "overlast"
 - "modelfout" if "case_piek" AND "case_blok" is "modelfout"
 - "plas" otherwise

NB: Some computation is done in RD, valid only in The Netherlands
"""

from __future__ import division
from __future__ import unicode_literals

import argparse
import logging
import numpy as np
import os
import sys


from osgeo import osr, ogr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from threedigrid.admin.gridresultadmin import GridH5ResultAdmin

logger = logging.getLogger(__name__)


GRIDADMIN_NAME = 'gridadmin.h5'
RESULTS_NAME = 'results_3di.nc'
POLYGON = 'POLYGON (({0} {1},{2} {1},{2} {3},{0} {3},{0} {1}))'


def filter_min_flow_area(lines, threshold, timestamp=None):
    """Filter lines below a given threshold in flow area (au)

    :param lines: GridH5ResultAdmin.lines to filter
    :param threshold: the minimum flow area, in square meters
    :param timestamp: the timestamp from which to take the flow areas from.
      Defaults to the last timestamp in the dataset.

    :returns: list of line ids with a gradient above or equal to threshold
    """
    if timestamp is None:
        timestamp = lines.timestamps[-1]
    data = lines.timeseries(
        start_time=timestamp,
        end_time=timestamp
    ).only('id', 'au').data
    return data['id'][(data['au'] >= threshold).any(axis=0)]


def filter_max_gradient(lines, nodes, threshold, timestamp=None):
    """Filter lines above a given threshold in waterlevel gradient

    The waterlevel gradient is computed by taking the difference of the
    waterlevel in the nodes that are connected by the line, and dividing by
    the length of the line. The length is computed by first projecting the line
    to EPSG28992, so this analysis is valid only in The Netherlands.

    :param lines: GridH5ResultAdmin.lines to filter
    :param nodes: GridH5ResultAdmin.nodes to get waterlevels (s1) from
    :param threshold: the maximum gradient, dimensionless (m / m)
    :param timestamp: the timestamp from which to take the waterlevels from.
      Defaults to the last timestamp in the dataset.

    :returns: list of line ids with a gradient below or equal to threshold
    """
    if timestamp is None:
        timestamp = lines.timestamps[-1]
    line_data = lines.timeseries(
        start_time=timestamp,
        end_time=timestamp
    ).only('id', 'line').data
    nodes_time = nodes.timeseries(start_time=timestamp, end_time=timestamp)
    nodes_line = nodes_time.filter(id__in=line_data['line'].ravel()).only(
        'id', 's1').data
    s1_first = nodes_line['s1'][0][np.searchsorted(
        nodes_line['id'],
        line_data['line'][0]
    )]
    s1_last = nodes_line['s1'][0][np.searchsorted(
        nodes_line['id'],
        line_data['line'][1]
    )]
    # compute length by transforming to RD. valid only in The Netherlands
    coords = lines.reproject_to(28992).line_coords
    length = np.sqrt(
        (coords[2] - coords[0]) ** 2 + (coords[3] - coords[1]) ** 2
    )
    mask = length == 0
    length[mask] = 1
    grad = (s1_last - s1_first) / length
    return line_data['id'][np.where(~mask & (np.abs(grad) <= threshold))[0]]


def filter_lines(gr, max_gradient, min_flow_area):
    """Filter lines from the gridresultadmin and return 3 filtered sets:

    :param gr: GridH5ResultAdmin
    :param max_gradient: GridH5ResultAdmin.nodes to get waterlevels (s1) from
    :param min_flow_area: the minimum flow area, in square meters

    :returns: tuple of 3 filtered GridH5ResultAdmin.lines objects:

        - 2D-2D lines that have flow_area >= min_flow_area and
          gradient <= max_gradient
        - 1D-2D lines that have flow_area >= min_flow_area
        - 1D-2D lines that have flow_area >= min_flow_area and
          gradient <= max_gradient
    """
    lines_active = filter_min_flow_area(gr.lines, min_flow_area)
    lines_valid = filter_max_gradient(gr.lines, gr.nodes, max_gradient)

    lines2d2d_valid = gr.lines.subset('2D_ALL').filter(
        id__in=np.intersect1d(lines_valid, lines_active)
    )
    lines1d2d_active = gr.lines.subset('1D2D').filter(
        id__in=lines_active
    )
    lines1d2d_valid = gr.lines.subset('1D2D').filter(
        id__in=np.intersect1d(lines_valid, lines_active)
    )
    return lines2d2d_valid, lines1d2d_active, lines1d2d_valid


def group_nodes(lines):
    """Group nodes into connected components

    :param lines: ndarray of shape (N, 2) with the ids of the connected nodes

    :returns: a list with group id per node: return_value[node_id] == group_id
    """
    coo = coo_matrix((np.ones(lines.shape[1]), lines),
                     shape=(lines.max() + 1,) * 2)

    # groups contains the ID of every group a node belongs to
    # there are many unconnected groups in there.
    _, groups = connected_components(coo, directed=False)

    return groups


def classify_nodes(node_id_2d, groups, lines1d2d_active, lines1d2d_valid):
    """Classify nodes in "plas", "overlast" or "modelfout"

    :param node_id_2d: a list of the ids of all 2D nodes
    :param groups: a list mapping groups[node_id] == group_id
    :param lines1d2d_active: a list of active 1D-2D lines (flow_area >=
      threshold)
    :param lines1d2d_valid: a list of valid 1D-2D lines (flow_area >= threshold
      and gradien <= threshold)

    :returns: 3 lists containing the node IDs of overlast, plas, and modelfout
    """
    plas_ids = []
    overlast_ids = []
    modelfout_ids = []

    # that have active 1D lines
    node_id_2d_active_1d = np.intersect1d(node_id_2d,
                                          lines1d2d_active.line.ravel())
    # that have valid 1D lines
    node_id_2d_valid_1d = np.intersect1d(node_id_2d_active_1d,
                                         lines1d2d_valid.line.ravel())
    # create boolean arrays for fast lookup
    is2d = np.zeros(node_id_2d.max() + 1, dtype=np.bool)
    is2d[node_id_2d] = True
    has1d_active = np.zeros_like(is2d)
    has1d_active[node_id_2d_active_1d] = True
    has1d_valid = np.zeros_like(is2d)
    has1d_valid[node_id_2d_valid_1d] = True

    for group_id in np.unique(groups):
        group_node_ids = np.where(groups == group_id)[0]
        group_is2d = is2d[group_node_ids]
        if len(group_is2d) == 1:
            # this could be 1D, check and if so, skip
            if not group_is2d[0]:
                continue
        else:
            # check, it should be 2D as we included only 2D2D lines in the
            # grouping
            if not group_is2d.all():
                raise RuntimeError("1D nodes should not occur")
        # if this group has a valid 1D line, categorize as overlast
        if has1d_valid[group_node_ids].any():
            overlast_ids.extend(group_node_ids.tolist())
        elif has1d_active[group_node_ids].any():
            modelfout_ids.extend(group_node_ids.tolist())
        else:
            plas_ids.extend(group_node_ids.tolist())

    return overlast_ids, plas_ids, modelfout_ids


def numpy_to_ogr_type(dtype):
    """Convert a numpy dtype to an ogr dtype (one of Real, Integer, or String)
    """
    if np.issubdtype(dtype, np.floating):
        return ogr.OFTReal
    elif np.issubdtype(dtype, np.integer):
        return ogr.OFTInteger
    else:
        return ogr.OFTString


def to_shape(cell_data, file_name, fields, epsg_code):
    """Write cell data to an ESRI shapefile.

    :param cell_data: dictionary containing 'cell_coords', 'id' and all fields
      the dictionary values are ndarrays with the same length (number of cells)
    :param file_name: the file name to output the file to. should not exist.
    :param epsg_code: the EPSG code of the projection
    """
    if fields is None:
        fields = ['id']
    fields = [str(f) for f in fields]
    for field in fields + ['cell_coords']:
        assert field in cell_data

    driver = ogr.GetDriverByName(str("ESRI Shapefile"))
    data_source = driver.CreateDataSource(str(file_name))

    sr = osr.SpatialReference()
    sr.ImportFromEPSG(int(epsg_code))
    layer = data_source.CreateLayer(
        str(os.path.basename(file_name)),
        sr,
        0
    )

    for field in fields:
        ogr_dtype = numpy_to_ogr_type(cell_data[field].dtype)
        layer.CreateField(ogr.FieldDefn(field, ogr_dtype))

    _definition = layer.GetLayerDefn()
    for i in range(cell_data['cell_coords'].shape[1]):
        feature = ogr.Feature(_definition)

        # get the cell coords
        cell_coords = [cell_data['cell_coords'][j][i] for j in range(4)]

        # create polygon and set it on the feature
        poly = ogr.CreateGeometryFromWkt(POLYGON.format(*cell_coords), sr)
        feature.SetGeometry(poly)

        # set fields
        for field in fields:
            feature.SetField(field, cell_data[field][i].item())
        layer.CreateFeature(feature)


def run_single(path, min_flow_area, max_gradient):
    logger.info("Analyzing scenario at {}".format(path))

    gr = GridH5ResultAdmin(os.path.join(path, GRIDADMIN_NAME),
                           os.path.join(path, RESULTS_NAME))

    lines2d2d_valid, lines1d2d_active, lines1d2d_valid = filter_lines(
        gr,
        min_flow_area=min_flow_area,
        max_gradient=max_gradient,
    )

    groups = group_nodes(lines2d2d_valid.line)

    cell_data = gr.cells.subset('2D_ALL').only("id", "cell_coords").data

    overlast_ids, plas_ids, modelfout_ids = classify_nodes(
        node_id_2d=cell_data['id'],
        groups=groups,
        lines1d2d_active=lines1d2d_active,
        lines1d2d_valid=lines1d2d_valid,
    )

    cell_data['case'] = np.full(cell_data['id'].size, '', dtype='S10')
    cell_data['case'][np.isin(cell_data['id'], plas_ids)] = 'plas'
    cell_data['case'][np.isin(cell_data['id'], overlast_ids)] = 'overlast'
    cell_data['case'][np.isin(cell_data['id'], modelfout_ids)] = 'modelfout'
    return cell_data, gr.epsg_code


def command(path_piek, path_blok, path_out,
            min_flow_area=0.001, max_gradient=0.00125):
    # check the existence of necessary files
    for path in (path_piek, path_blok):
        if not os.path.isdir(path):
            raise IOError('{} does not exist'.format(path))
        if not os.path.isfile(os.path.join(path, GRIDADMIN_NAME)):
            raise IOError('{} does not exist in {}'.format(
                GRIDADMIN_NAME, path))
        if not os.path.isfile(os.path.join(path, RESULTS_NAME)):
            raise IOError('{} does not exist in {}'.format(
                RESULTS_NAME, path))

    # check if the output file does not exist, but its directory should exist
    if not os.path.isdir(os.path.dirname(path_out)):
        raise IOError('{} does not exist'.format(os.path.dirname(path_out)))
    if os.path.isfile(path_out):
        raise IOError('{} already exists'.format(path_out))

    # run the analyses
    cell_data_piek, epsg = run_single(path_piek, min_flow_area, max_gradient)
    cell_data_blok, epsg = run_single(path_blok, min_flow_area, max_gradient)

    # check if the cell coords are precisely equal
    comp = cell_data_piek['cell_coords'] == cell_data_blok['cell_coords']
    if not comp.all():
        raise RuntimeError("Blok and Piek scenarios have unequal cell coords")

    # assemble the data
    cell_data = {
        'id': cell_data_piek['id'],
        'cell_coords': cell_data_piek['cell_coords'],
        'case_piek': cell_data_piek['case'],
        'case_blok': cell_data_blok['case'],
    }

    # logical operations to generate "case_final"
    cell_data['case_final'] = np.full(cell_data['id'].size, '', dtype='S10')
    cell_data['case_final'][
        (cell_data['case_blok'] == 'plas')
        | (cell_data['case_piek'] == 'plas')
    ] = 'plas'
    cell_data['case_final'][
        (cell_data['case_blok'] == 'overlast')
        | (cell_data['case_piek'] == 'overlast')
    ] = 'overlast'
    cell_data['case_final'][
        (cell_data['case_blok'] == 'modelfout')
        & (cell_data['case_piek'] == 'modelfout')
    ] = 'modelfout'

    logger.info("Writing shapefile at {}...".format(path_out))
    to_shape(
        cell_data,
        path_out,
        fields=['id', 'case_blok', 'case_piek', 'case_final'],
        epsg_code=epsg,
    )
    logger.info("Done.")


def get_parser():
    """
    Compute a shapefile that distinguishes "plas", "overlast", and "modelfout"
    from two 3Di gridadmin and results files ("piek" and "blok").
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'path_piek',
        help=(
            'Path to 3Di results that contain the "piek" simulation. The '
            'files {} and {} have to be present in this directory.'.format(
                RESULTS_NAME, GRIDADMIN_NAME,
            )
        )
    )
    parser.add_argument(
        'path_blok',
        help=(
            'Path to 3Di results that contain the "blok" simulation. The '
            'files {} and {} have to be present in this directory.'.format(
                RESULTS_NAME, GRIDADMIN_NAME
            )
        )
    )
    parser.add_argument(
        'path_out',
        help='File to output the resulting shapefile. Should not exist.',
    )
    parser.add_argument(
        '-f', '--flow_area',
        default=0.001,
        dest='min_flow_area',
        help=('Minimum flow area ("doorstroomoppervlak") in square meters '
              'for node connections to be included in this analysis.'),
        type=float,
    )
    parser.add_argument(
        '-g', '--gradient',
        default=0.00125,
        dest='max_gradient',
        help=('Maximum gradient ("verhang") (no units, lengh per length) '
              'for node connections to be valid in this analysis.'),
        type=float,
    )
    return parser


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format='%(message)s')

    command(**vars(get_parser().parse_args()))
