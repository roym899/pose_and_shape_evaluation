import os.path as osp
import numpy as np
import numpy.linalg as LA
import copy
import random

import open3d as o3
import torch

from . import common3Dfunc as c3D
from .asm_pcd import asm
from .ASM_Net import pointnet

"""
  Path setter
"""


def set_paths(dataset_root, category):

    paths = {}
    paths["trainset_path"] = osp.join(dataset_root, category, "train")
    """
    paths["testset_path"] = osp.join(dataset_root,category,"test")
    paths["valset_path"] = osp.join(dataset_root,category,"val")
    paths["original_path"] = osp.join(dataset_root,category,"original")
    paths["sorted_path"] = osp.join(dataset_root,category,"sorted")
    paths["trainmodels_path"] = osp.join(dataset_root,category,"train_models")
    paths["testmodels_path"] = osp.join(dataset_root,category,"test_models")
    paths["valmodels_path"] = osp.join(dataset_root,category,"val_models")
    """

    for p in paths.values():
        if osp.exists(p) is not True:
            print("!!ERROR!! Path not found. Following path is not found.")
            print(p)
            return False

    return paths


def load_asmds(root, synset_names):
    """load multiple Active Shape Model Deformations
    Args:
      root(str): Root directory
      synset_names(str):　List of class names.
                          The first element "BG" is ignored.
    Return:
      dict: A dictionary of ASMDeformation
    """
    # print("Root dir:", root)
    asmds = {}
    for s in range(len(synset_names) - 1):
        paths = set_paths(root, synset_names[s + 1])
        trainset_path = paths["trainset_path"]
        info = np.load(osp.join(trainset_path, "info.npz"))
        asmd = asm.ASMdeformation(info)
        asmds[synset_names[s + 1]] = asmd

    return asmds


def load_models(root, dirname, n_epoch, synset_names, ddim, n_points, device):
    """Load multiple network weights (for experiments)
    Args:
      root(str):　Path to dataset root
      dirname(str): Directory name of weights
      n_epoch(int): choose the epoch of weights
      synset_names(str):　The first element is "BG" should be ignored.
      use_dim(int): # of dimensions used to deformation
      n_points(int): # of points fed to the networks
      device(str): device("cuda:0" or "cpu")
    Return:
      A dictionary of weights
    """
    # print("Root dir:", root)
    models = {}
    for s in range(len(synset_names) - 1):
        path = osp.join(
            root,
            synset_names[s + 1],
            "weights",
            dirname,
            "model_" + str(n_epoch) + ".pth",
        )
        print(" loading:", path)

        total_dim = ddim + 1  # deformation(ddim) + scale(1)
        model = pointnet.ASM_Net(k=total_dim, num_points=n_points)
        model.load_state_dict(torch.load(path))
        model.to(device)
        model.eval()
        models[synset_names[s + 1]] = model

    return models


def load_models_release(root, synset_names, ddim, n_points, device):
    """Load multiple network weights (for release)
    Args:
      root(str):　Path to model root
      synset_names(str):　The first element is "BG" should be ignored.
      use_dim(int): # of dimensions used to deformation
      n_points(int): # of points fed to the networks
      device(str): device("cuda:0" or "cpu")
    Return:
      A dictionary of weights
    """
    # print("Root dir:", root)
    models = {}
    for s in range(len(synset_names) - 1):
        path = osp.join(root, synset_names[s + 1], "model.pth")
        # print(" loading:", path)

        total_dim = ddim + 1  # deformation(ddim) + scale(1)
        model = pointnet.ASM_Net(k=total_dim, num_points=n_points)
        model.load_state_dict(torch.load(path))
        model.to(device)
        model.eval()
        models[synset_names[s + 1]] = model

    return models


def get_pcd_from_rgbd(im_c, im_d, intrinsic):
    """generate point cloud from cv2 image

    Args:
      im_c(ndarray 3ch): RGB image
      im_d(ndarray 1ch): Depth image
      intrinsic(PinholeCameraIntrinsic): intrinsic parameter
    Return:
      open3d.geometry.PointCloud: point cloud
    """
    color_raw = o3.geometry.Image(im_c)
    depth_raw = o3.geometry.Image(im_d)

    rgbd_image = o3.geometry.RGBDImage.create_from_color_and_depth(
        color_raw,
        depth_raw,
        depth_scale=1000.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False,
    )
    pcd = o3.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    return pcd


def random_sample(data, n_sample):

    if n_sample < data.shape[0]:
        choice = random.sample(list(np.arange(0, data.shape[0], 1)), k=n_sample)
    else:
        choice = random.choices(list(np.arange(0, data.shape[0], 1)), k=n_sample)

    sampled = np.array(data[choice])
    return copy.deepcopy(sampled)


def generate_pose():
    """generate pose from hemisphere-distributed viewpoints"""

    # y axis(yr): -pi - pi
    # x axis(xr): 0 - 0.5pi
    # view_direction(ar): -0.1pi - 0.1pi
    yr = (random.random() * 2.0 * np.pi) - np.pi
    xr = random.random() * 0.5 * np.pi
    ar = (random.random() * 0.2 * np.pi) - (0.1 * np.pi)

    # x,y-axis
    y = c3D.RPY2Matrix4x4(0, yr, 0)[:3, :3]
    x = c3D.RPY2Matrix4x4(xr, 0, 0)[:3, :3]
    rot = np.dot(x, y)

    # rotation around view axis
    v = np.array([0.0, 0.0, -1.0])  # basis vector
    rot_v = np.dot(x, v)  # prepare axis
    q = np.hstack([ar, rot_v])  # generate quaternion
    q = q / LA.norm(q)  # unit quaternion

    pose = c3D.quaternion2rotation(q)
    rot = np.dot(pose, rot)

    return rot


def get_mask(mask_info, choice="pred"):
    """
    Args:
      mask_info(dict): object mask of "GT" and "Mask RCNN used NOCS_CVPR2019)
      choice(str): choice of mask．gt(GT) or pred(Mask-RCNN).
    Return:
      tuple: mask
    """
    key_id = choice + "_class_ids"
    key_mask = choice + "_masks"

    class_ids = mask_info[key_id]
    mask = mask_info[key_mask]

    return np.asarray(mask), np.asarray(class_ids)


def get_model_scale(image_path, model_root):
    model_path = None
    meta_path = image_path + "_meta.txt"
    sizes = []
    class_ids = []
    pcds = []
    with open(meta_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        words = line[:-1].split(" ")
        model_path = osp.join(model_root, words[-1] + ".obj")
        pcd = o3.io.read_triangle_mesh(model_path)
        bb = pcd.get_axis_aligned_bounding_box()
        bbox = bb.get_max_bound() - bb.get_min_bound()
        size = np.linalg.norm(bbox)
        sizes.append(size)
        class_ids.append(int(words[1]))
        pcds.append(pcd)

    return np.asarray(sizes), np.asarray(class_ids), pcds


def quaternion2rotationPT( q ):
    """ Convert unit quaternion to rotation matrix
    
    Args:
        q(torch.tensor): unit quaternion (N,4), scalar first
    Returns:
        torch.tensor: rotation matrix (N,3,3)
    """
    r11 = (q[:,0]**2+q[:,1]**2-q[:,2]**2-q[:,3]**2).unsqueeze(0).T
    r12 = (2.0*(q[:,1]*q[:,2]-q[:,0]*q[:,3])).unsqueeze(0).T
    r13 = (2.0*(q[:,1]*q[:,3]+q[:,0]*q[:,2])).unsqueeze(0).T

    r21 = (2.0*(q[:,1]*q[:,2]+q[:,0]*q[:,3])).unsqueeze(0).T
    r22 = (q[:,0]**2+q[:,2]**2-q[:,1]**2-q[:,3]**2).unsqueeze(0).T
    r23 = (2.0*(q[:,2]*q[:,3]-q[:,0]*q[:,1])).unsqueeze(0).T

    r31 = (2.0*(q[:,1]*q[:,3]-q[:,0]*q[:,2])).unsqueeze(0).T
    r32 = (2.0*(q[:,2]*q[:,3]+q[:,0]*q[:,1])).unsqueeze(0).T
    r33 = (q[:,0]**2+q[:,3]**2-q[:,1]**2-q[:,2]**2).unsqueeze(0).T
    
    r = torch.cat( (r11,r12,r13,
                r21,r22,r23,
                r31,r32,r33), 1 )
    r = torch.reshape( r, (q.shape[0],3,3))
    
    return r