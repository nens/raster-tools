# -*- coding: utf-8 -*-
"""
Fill in the interior of some masked void in a predictable way.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import collections
import math

import numpy as np
import potrace

from scipy.ndimage import binary_erosion, binary_dilation


np.set_printoptions(threshold=2500, linewidth=200)


def interior(mask, size):
    """
    Return numpy index tuple for interior points inside mask.
    Uses morphology for shrinking, kdtree for pair selection. Make a
    chain and select points. Repeat.
    """
    mask = binary_erosion(mask, iterations=size).astype('u1')
    print(mask)
    bitmap = potrace.Bitmap(mask)
    path = bitmap.trace()
    print(path.curves)

    
    links = collections.defaultdict(list)
    
    # selection
    # m = 9; a = np.zeros(len(points), 'u1')
    count = len(points)
    index = np.arange(0, count, count / np.ceil(count / size)).astype('u8')
    result.extend(points[index].tolist())
    points[np.arange(0, m, m / np.ceil(m / n)).astype('u8')] = 1; a

z = np.zeros((49, 49), 'u1')
x, y = np.indices(z.shape) - 20
z[np.sqrt(x ** 2 + y ** 2) < 20] = 1
z[21:48, 21:48] = 1



# print(z)
interior(mask=z, size=19)



