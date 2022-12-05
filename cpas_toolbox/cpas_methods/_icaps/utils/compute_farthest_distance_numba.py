import math

import numpy as np

# import pymesh
# from numba import jit


# @jit(nopython=True)
# def get_dist_sq(pt1, pt2):
#     return (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2 + (pt1[2] - pt2[2]) ** 2


# @jit(nopython=True)
# def get_max_dist(vertices, num_points):
#     max_dist = -1
#     ind = [-1, -1]
#     for v1 in range(num_points - 1):
#         for v2 in range(v1 + 1, num_points):
#             dist_sq = get_dist_sq(vertices[v1, :], vertices[v2, :])
#             if math.sqrt(dist_sq) > max_dist:
#                 ind[0] = v1
#                 ind[1] = v2
#                 max_dist = math.sqrt(dist_sq)
#     return ind, max_dist


def get_bbox_dist(vertices):
    max_x = max(vertices[:, 0])
    max_y = max(vertices[:, 1])
    max_z = max(vertices[:, 2])

    min_x = min(vertices[:, 0])
    min_y = min(vertices[:, 1])
    min_z = min(vertices[:, 2])

    dist_sq = (max_x - min_x) ** 2 + (max_y - min_y) ** 2 + (max_z - min_z) ** 2

    return math.sqrt(dist_sq)
