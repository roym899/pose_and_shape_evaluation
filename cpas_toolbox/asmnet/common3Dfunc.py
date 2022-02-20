# Common functions for 3D processiong
# Shuichi Akizuki, Chukyo Univ.
# Email: s-akizuki@sist.chukyo-u.ac.jp
#

import open3d as o3
import cv2
import numpy as np
import numpy.linalg as LA
import copy
import json
from math import *


class Mapping:
    def __init__(self, camera_intrinsic_name, _w=640, _h=480, _d=1000.0):
        self.camera_intrinsic = o3.io.read_pinhole_camera_intrinsic(
            camera_intrinsic_name
        )
        self.width = _w
        self.height = _h
        self.d = _d
        self.camera_intrinsic4x4 = np.identity(4)
        self.camera_intrinsic4x4[0, 0] = self.camera_intrinsic.intrinsic_matrix[0, 0]
        self.camera_intrinsic4x4[1, 1] = self.camera_intrinsic.intrinsic_matrix[1, 1]
        self.camera_intrinsic4x4[0, 3] = self.camera_intrinsic.intrinsic_matrix[0, 2]
        self.camera_intrinsic4x4[1, 3] = self.camera_intrinsic.intrinsic_matrix[1, 2]

    def showCameraIntrinsic(self):
        print(self.camera_intrinsic.intrinsic_matrix)
        print(self.camera_intrinsic4x4)

    def Cloud2Image(self, cloud_in, dbg_vis=False):

        img = np.zeros([self.height, self.width], dtype=np.uint8)
        img_zero = np.zeros([self.height, self.width], dtype=np.uint8)

        cloud_np1 = np.asarray(cloud_in.points)
        sorted_indices = np.argsort(cloud_np1[:, 2])[::-1]
        cloud_np = cloud_np1[sorted_indices]
        cloud_np_xy = cloud_np[:, 0:2] / cloud_np[:, [2]]
        # cloud_np ... (x/z, y/z, z)
        cloud_np = np.hstack((cloud_np_xy, cloud_np[:, [2]]))

        cloud_color1 = np.asarray(cloud_in.colors)

        cloud_mapped = o3.geometry.PointCloud()
        cloud_mapped.points = o3.utility.Vector3dVector(cloud_np)

        cloud_mapped.transform(self.camera_intrinsic4x4)

        """ If cloud_in has the field of color, color is mapped into the image. """
        if len(cloud_color1) == len(cloud_np):
            cloud_color = cloud_color1[sorted_indices]
            img = cv2.merge((img, img, img))
            for i, pix in enumerate(cloud_mapped.points):
                if (
                    pix[0] < self.width
                    and 0 < pix[0]
                    and pix[1] < self.height
                    and 0 < pix[1]
                ):
                    img[int(pix[1]), int(pix[0])] = (cloud_color[i] * 255.0).astype(
                        np.uint8
                    )
                    # color = (cloud_color[i]*255.0)
                    # cv2.circle(img, (int(pix[0]),int(pix[1])), 1, (int(color[0]),int(color[1]),int(color[2])), -1, cv2.LINE_AA )

            img = img[:, :, (2, 1, 0)]

        else:
            d_max = cloud_np[0, 2]
            d_min = cloud_np[-1, 2]
            base = d_max - d_min
            if base == 0:
                img = cv2.merge((img_zero, img_zero, img_zero))
                return img
            for i, pix in enumerate(cloud_mapped.points):
                if (
                    pix[0] < self.width
                    and 0 < pix[0]
                    and pix[1] < self.height
                    and 0 < pix[1]
                ):
                    img[int(pix[1]), int(pix[0])] = int(
                        255.0 * ((d_max - cloud_np[i, 2]) / base)
                    )
            img = cv2.merge((img_zero, img, img_zero))

            if dbg_vis:
                for i, pix in enumerate(cloud_mapped.points):
                    value = int(
                        np.clip(200.0 * ((d_max - cloud_np[i, 2]) / base), 0, 255)
                    )
                    cv2.circle(
                        img,
                        (int(pix[0]), int(pix[1])),
                        1,
                        (0, value, 0),
                        -1,
                        cv2.LINE_AA,
                    )

        return img

    def Pix2Pnt(self, pix, val):
        pnt = np.array([0.0, 0.0, 0.0], dtype=np.float)
        depth = val / self.d
        # print('[0,2]: {}'.format(self.camera_intrinsic.intrinsic_matrix[0,2]))
        # print('[1,2]: {}'.format(self.camera_intrinsic.intrinsic_matrix[1,2]))
        # print(self.camera_intrinsic.intrinsic_matrix)
        pnt[0] = (
            (float(pix[0]) - self.camera_intrinsic.intrinsic_matrix[0, 2])
            * depth
            / self.camera_intrinsic.intrinsic_matrix[0, 0]
        )
        pnt[1] = (
            (float(pix[1]) - self.camera_intrinsic.intrinsic_matrix[1, 2])
            * depth
            / self.camera_intrinsic.intrinsic_matrix[1, 1]
        )
        pnt[2] = depth

        return pnt


def centering(_cloud_in):
    """
    Centering()
    offset an input cloud to its centroid.

    input(s):
        _cloud_in: point cloud to be centered
    output(s):
        cloud_off:
        center:
    """
    cloud_in = copy.deepcopy(_cloud_in)
    np_m = np.asarray(cloud_in.points)
    center = np.mean(np_m, axis=0)
    np_m[:] -= center

    cloud_off = o3.geometry.PointCloud()
    cloud_off.points = o3.utility.Vector3dVector(np_m)

    return cloud_off, center


def size_normalization(pcd):
    """
    point cloud size normalization

    input(s):
        pcd: point cloud to be scaled
    output(s):
        pcd_scaled: scaled  point cloud
    """
    np_pcd = np.asarray(pcd.points)
    np_length = np.linalg.norm(np_pcd, axis=1)
    max_size = np.max(np_length)
    np_pcd_scaled = np_pcd / max_size
    pcd_scaled = o3.geometry.PointCloud()
    pcd_scaled.points = o3.utility.Vector3dVector(np_pcd_scaled)

    return pcd_scaled, max_size


def ComputeTransformationMatrixAroundCentroid(_cloud_in, _roll, _pitch, _yaw):

    """offset center"""
    np_in = np.asarray(_cloud_in.points)
    center = np.mean(np_in, axis=0)
    offset = np.identity(4)
    offset[0:3, 3] -= center

    """ rotation """
    rot = RPY2Matrix4x4(_roll, _pitch, _yaw)

    """ reverse offset """
    reverse = np.identity(4)
    reverse[0:3, 3] = center

    final = np.dot(reverse, np.dot(rot, offset))

    return final


def Scaling(cloud_in, scale):
    """
    multiply scaling factor to the input point cloud.
    input(s):
        cloud_in: point cloud to be scaled.
        scale: scaling factor
    output(s):
        cloud_out:
    """
    cloud_np = np.asarray(cloud_in.points)
    cloud_np *= scale
    cloud_out = o3.PointCloud()
    cloud_out.points = o3.Vector3dVector(cloud_np)

    return cloud_out


def Offset(cloud_in, offset):
    cloud_np = np.asarray(cloud_in.points)
    cloud_np += offset
    cloud_off = o3.PointCloud()
    cloud_off.points = o3.Vector3dVector(cloud_np)
    return cloud_off


def translation_jitter(pcd, _scale):
    """Apply translation jitter to an point cloud
        as noise
    Args:
      pcd(open3d.geometry.PointCloud): input point cloud
      _scale(array_like of floats): standard divitation of translation

    Return:
      open3d.geometry.PointCloud: output point cloud
    """

    pcd_out = copy.deepcopy(pcd)
    tj = np.random.normal(loc=[0.0, 0.0, 0.0], scale=_scale, size=3)
    pcd_out.translate(tj)
    return pcd_out


def RPY2Matrix4x4(roll, pitch, yaw):

    rot = np.identity(4)
    if roll < -3.141:
        roll += 6.282
    elif 3.141 < roll:
        roll -= 6.282
    if pitch < -3.141:
        pitch += 6.282
    elif 3.141 < pitch:
        pitch -= 6.282
    if yaw < -3.141:
        yaw += 6.282
    elif 3.141 < yaw:
        yaw -= 6.282

    rot[0, 0] = cos(yaw) * cos(pitch)
    rot[0, 1] = -sin(yaw) * cos(roll) + (cos(yaw) * sin(pitch) * sin(roll))
    rot[0, 2] = sin(yaw) * sin(roll) + (cos(yaw) * sin(pitch) * cos(roll))
    rot[1, 0] = sin(yaw) * cos(pitch)
    rot[1, 1] = cos(yaw) * cos(roll) + (sin(yaw) * sin(pitch) * sin(roll))
    rot[1, 2] = -cos(yaw) * sin(roll) + (sin(yaw) * sin(pitch) * cos(roll))
    rot[2, 0] = -sin(pitch)
    rot[2, 1] = cos(pitch) * sin(roll)
    rot[2, 2] = cos(pitch) * cos(roll)
    rot[3, 0] = rot[3, 1] = rot[3, 2] = 0.0
    rot[3, 3] = 1.0

    return rot


"""Convert Rotation matrix to RPY parameters"""


def Mat2RPY(rot):
    roll = atan2(rot[2, 1], rot[2, 2])
    pitch = atan2(-rot[2, 0], sqrt(rot[2, 1] * rot[2, 1] + rot[2, 2] * rot[2, 2]))
    yaw = atan2(rot[1, 0], rot[0, 0])

    return roll, pitch, yaw


def makeTranslation4x4(offset):

    trans = np.identity(4)
    trans[0:3, 3] = offset

    return trans


""" save transformation matrix as a .json file. """


def save_transformation(trans, name):
    trans_list = trans.tolist()
    transform = {"transformation4x4": [trans_list]}
    f_out = open(name, "w")
    json.dump(transform, f_out)


""" load transformation matrix from a .json file. """


def load_transformation(name):
    f_in = open(name, "r")
    json_data = json.load(f_in)
    trans_in = np.array(json_data["transformation4x4"][0])

    return trans_in


def calc_mesh_resolution(pcd):
    nearest_dist = 0
    valid = 0
    pcd_tree = o3.geometry.KDTreeFlann(pcd)
    for t in pcd.points:
        [k, idx, d] = pcd_tree.search_knn_vector_3d(t, 2)
        if d[1] != 0.0:
            nearest_dist += d[1]
            valid += 1

    return np.sqrt(nearest_dist / valid)


def transformation_error(rt1, rt2):
    """
    rt1, rt2: 4x4 matrix

    return
       angular_error: angular error in degree
       translation_error: translation error
    """
    # angular error
    rotation1 = rt1[:3, :3]
    rotation2 = rt2[:3, :3]
    p = np.array([1.0, 0.0, 0.0])
    p1 = np.dot(rotation1, p)
    p2 = np.dot(rotation2, p)
    angular_error = np.degrees(np.arccos(np.dot(p1, p2)))

    # translation error
    translation1 = rt1[:3, 3]
    translation2 = rt2[:3, 3]
    diff = translation2 - translation1
    translation_error = LA.norm(diff, ord=2, axis=0)

    return angular_error, translation_error


def quaternion2rotation(q):
    """Convert unit quaternion to rotation matrix

    Args:
        q(array like): unit quaterinon

    Returns:
        ndarray: rotation matrix 3x3
    """
    rot = np.array(
        [
            [
                q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2,
                2.0 * (q[1] * q[2] - q[0] * q[3]),
                2.0 * (q[1] * q[3] + q[0] * q[2]),
            ],
            [
                2.0 * (q[1] * q[2] + q[0] * q[3]),
                q[0] ** 2 + q[2] ** 2 - q[1] ** 2 - q[3] ** 2,
                2.0 * (q[2] * q[3] - q[0] * q[1]),
            ],
            [
                2.0 * (q[1] * q[3] - q[0] * q[2]),
                2.0 * (q[2] * q[3] + q[0] * q[1]),
                q[0] ** 2 + q[3] ** 2 - q[1] ** 2 - q[2] ** 2,
            ],
        ]
    )
    return rot


def z_quantization(pcd, reso):
    """点群をz軸方向に指定したステップ数で量子化する

    Args:
      pcd(open3d.geometry.PointCloud): input point cloud
      reso(int): z-resolution
    Returns:
      open3d.geometry.PointCloud: quantized point cloud
    """
    np_pcd = np.asarray(pcd.points).copy()
    max_z = np.max(np_pcd, axis=0)[2]
    min_z = np.min(np_pcd, axis=0)[2]
    range_z = max_z - min_z
    unit = range_z / reso
    np_pcd[:, 2] = (np_pcd[:, 2] // unit) * unit
    pcd_q = o3.geometry.PointCloud()
    pcd_q.points = o3.utility.Vector3dVector(np_pcd)
    return pcd_q


def merge_pcds(pcds):
    """Merge point clouds
    Args:
      pcds(list): A list of o3.geometry.PointCloud
    Return:
      o3.geometry.PointCloud: merged point cloud
    """
    print(pcds)
    np_pcd = []
    for p in pcds:
        np_pcd.append(np.asarray(p.points))
    np_pcd = np.vstack(np_pcd)

    pcd_merged = o3.geometry.PointCloud()
    pcd_merged.points = o3.utility.Vector3dVector(np_pcd)
    return pcd_merged


def image_statistical_outlier_removal(im, factor=2.0):
    """Outlier removal on depth image
    Args:
        im(numpy.ndarray): depth image
        factor(float): threshold
    """
    np_nonzero = np.nonzero(im)
    depth = im[np_nonzero]
    mean = np.mean(depth)
    std = np.std(depth)

    threshold = mean + factor * std
    im_inlier_mask = np.where(im < threshold, 1, 0)
    im_clean = im * im_inlier_mask

    kernel = np.ones((5, 5), np.uint8)
    im_bin = np.where(0 < im_clean, 1, 0).astype(np.uint8)
    im_bin = cv2.morphologyEx(im_bin, cv2.MORPH_OPEN, kernel)
    im_clean2 = im_clean * im_bin.astype(im.dtype)

    return im_clean.astype(np.uint16)


def applyHPR(pcd, viewpoint=np.array([0.0, 0.0, 0.0]), hpr_param=50):
    out_hpr = pcd.hidden_point_removal(viewpoint, hpr_param)
    points = out_hpr[0].vertices
    visible_pcd = o3.geometry.PointCloud()
    visible_pcd.points = points
    return visible_pcd
