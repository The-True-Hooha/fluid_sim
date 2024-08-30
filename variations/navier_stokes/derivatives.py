from functools import reduce
from itertools import cycle
from math import factorial
import numpy as np
import scipy.sparse as sp

# get the finite difference between the surfaces
def coefficient_difference(d, accuracy = 1):
    d += 1
    radius = accuracy + d // 2 - 1
    points = range(-radius, radius + 1)
    coeff = np.linalg.inv(np.vander(points))
    return coeff[-d] * factorial(d - 1), points


def operators(shape, *differences):
    differences = zip(shape, cycle(differences))
    factors = (sp.diags(*diff, shape=(dim, ) * 2) for dim, diff in differences)
    return reduce(lambda a, f: sp.kronsum(f, a, format='csc'), factors)
    