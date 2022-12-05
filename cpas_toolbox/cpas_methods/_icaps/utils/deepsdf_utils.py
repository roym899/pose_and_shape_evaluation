import os
import sys

import cv2
import numpy as np
import numpy.ma as ma
import torch

from ..datasets.nocs_real_dataset import *
from ..deep_sdf.deep_sdf_decoder import *
from ..deep_sdf.deepsdf_optim import *
from ..deep_sdf.evaluator import *
from .decoder_utils import *

# from chamferdist import ChamferDistance


def vis_single(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    display_interval = 20
    ax.scatter(
        points[::display_interval, 0],
        points[::display_interval, 1],
        points[::display_interval, 2],
        color="red",
    )
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    min_coor = -1
    max_coor = 1
    ax.set_xlim(min_coor, max_coor)
    ax.set_ylim(min_coor, max_coor)
    ax.set_zlim(min_coor, max_coor)
    plt.show()


def Twc_np(pose):

    Twc = np.zeros((4, 4), dtype=np.float32)

    Twc[:3, :3] = quat2mat(pose[3:])
    Twc[:3, 3] = pose[:3]
    Twc[3, 3] = 1

    return Twc


def depth2pc(depth, h, w, intrinsics):

    ymap = np.array([[j for i in range(w)] for j in range(h)])
    xmap = np.array([[i for i in range(w)] for j in range(h)])

    pt2 = depth
    pt0 = (xmap - intrinsics[0, 2]) * pt2 / intrinsics[0, 0]
    pt1 = (ymap - intrinsics[1, 2]) * pt2 / intrinsics[1, 1]

    mask_depth = ma.getmaskarray(ma.masked_greater(pt2, 0))
    mask = mask_depth

    choose = mask.flatten().nonzero()[0]

    pt2_valid = pt2.flatten()[choose][:, np.newaxis].astype(np.float32)
    pt0_valid = pt0.flatten()[choose][:, np.newaxis].astype(np.float32)
    pt1_valid = pt1.flatten()[choose][:, np.newaxis].astype(np.float32)

    ps_c = np.concatenate(
        (pt0_valid, pt1_valid, pt2_valid, np.ones_like(pt0_valid)), axis=1
    )
    return ps_c


def visualize_pose_comparison(points_c_meas, points_all_np, T_co_init, T_co_opt):
    # visualization for debugging
    points_init = np.matmul(
        np.linalg.inv(T_co_init), points_c_meas.cpu().numpy().transpose()
    ).transpose()
    points_opt = np.matmul(
        np.linalg.inv(T_co_opt), points_c_meas.cpu().numpy().transpose()
    ).transpose()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    display_interval = 20
    ax.scatter(
        points_all_np[::display_interval, 0],
        points_all_np[::display_interval, 1],
        points_all_np[::display_interval, 2],
        color="green",
    )
    ax.scatter(
        points_init[::display_interval, 0],
        points_init[::display_interval, 1],
        points_init[::display_interval, 2],
        color="red",
    )
    ax.scatter(
        points_opt[::display_interval, 0],
        points_opt[::display_interval, 1],
        points_opt[::display_interval, 2],
        color="blue",
    )
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    min_coor = -0.2
    max_coor = 0.2
    ax.set_xlim(min_coor, max_coor)
    ax.set_ylim(min_coor, max_coor)
    ax.set_zlim(min_coor, max_coor)
    plt.show()


def visualize_shape_comparison(
    points_gt, latent_gt, latent_tensor, latent_tensor_opt, evaluator, decoder, size_est
):
    fname = os.path.join("./vis/", "mesh.ply")
    points_init = None
    points_opt = None
    dist_init = 10000
    dist_opt = 10000
    try:
        points_init = evaluator.latent_vec_to_points(
            latent_tensor, num_points=10000, silent=True
        )
        points_opt = evaluator.latent_vec_to_points(
            latent_tensor_opt, num_points=10000, silent=True
        )
    # points_gt_sdf = evaluator.latent_vec_to_points(latent_gt, num_points=10000, silent=True)
    except:
        print("latent to points failed...")
        return dist_init, dist_opt, points_init, points_opt
    if points_init is None or points_opt is None:
        return dist_init, dist_opt, points_init, points_opt
    points_init_metrics = points_init.astype(np.float32) * (size_est)
    points_opt_metrics = points_opt.astype(np.float32) * (size_est)
    # compute chamfer distance
    try:
        dist_init = compute_chamfer_dist(points_init_metrics, points_gt.float().numpy())
        dist_opt = compute_chamfer_dist(points_opt_metrics, points_gt.float().numpy())
    except:
        print("compute chamfer failed...")

    print("initial chamfer distance = {}".format(dist_init))
    print("refined chamfer distance = {}".format(dist_opt))
    print("refined - initial chamfer distance = {}".format(dist_opt - dist_init))

    return (
        dist_init,
        dist_opt,
        points_init_metrics,
        points_opt_metrics,
    )  # dist_init, dist_opt, sdf_dist_init, sdf_dist_opt, sdf_dist_gt, points_init, points_opt, points_gt_sdf


def visualize_depth_pc(points, points_target):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    display_interval = 10
    ax.scatter(
        points[::display_interval, 0],
        points[::display_interval, 1],
        points[::display_interval, 2],
        color="green",
    )
    ax.scatter(
        points_target[::display_interval, 0],
        points_target[::display_interval, 1],
        points_target[::display_interval, 2],
        color="red",
    )

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    plt.show()


def visualize_pn_training_data(points, points_downsample, label_gt, evaluator):
    fname = os.path.join("./vis/", "mesh.ply")
    points_from_label_gt = evaluator.latent_vec_to_points(
        label_gt, num_points=10000, silent=True
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    display_interval = 20
    ax.scatter(
        points_downsample[::display_interval, 0],
        points_downsample[::display_interval, 1],
        points_downsample[::display_interval, 2],
        color="green",
    )
    ax.scatter(
        points[::display_interval, 0],
        points[::display_interval, 1],
        points[::display_interval, 2],
        color="red",
    )
    ax.scatter(
        points_from_label_gt[::display_interval, 0],
        points_from_label_gt[::display_interval, 1],
        points_from_label_gt[::display_interval, 2],
        color="blue",
    )
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    min_coor = -1
    max_coor = 1
    ax.set_xlim(min_coor, max_coor)
    ax.set_ylim(min_coor, max_coor)
    ax.set_zlim(min_coor, max_coor)
    plt.show()


def compute_chamfer_dist(points1_np, points2_np):
    points1_torch = torch.from_numpy(points1_np).unsqueeze(0)
    points2_torch = torch.from_numpy(points2_np).unsqueeze(0)

    chamferDist = ChamferDistance()

    dist_bidirectional = (
        chamferDist(points1_torch, points2_torch, bidirectional=True)
        .detach()
        .cpu()
        .item()
    )

    return dist_bidirectional / (points1_torch.size(1) + points2_torch.size(1))


def decode_sdf(
    decoder, latent_vector, points, clamp_dist=0.1, MAX_POINTS=100000, no_grad=False
):
    start, num_all = 0, points.shape[0]
    output_list = []
    while True:
        end = min(start + MAX_POINTS, num_all)
        if latent_vector is None:
            inputs = points[start:end]
        else:
            latent_repeat = latent_vector.expand(end - start, -1)
            inputs = torch.cat([latent_repeat, points[start:end]], 1)
        sdf_batch = decoder.inference(inputs)
        start = end
        if no_grad:
            sdf_batch = sdf_batch.detach()
        output_list.append(sdf_batch)
        if end == num_all:
            break
    sdf = torch.cat(output_list, 0)

    if clamp_dist != None:
        sdf = torch.clamp(sdf, -clamp_dist, clamp_dist)

    loss_fun = nn.L1Loss(reduction="mean")
    loss_fun = loss_fun.cuda()
    dist_target = torch.zeros_like(sdf)
    dist_target = dist_target.cuda()
    loss = max(abs(sdf))
    return loss.data.cpu().item()


def visualize_shape_pc(points_gt_sdf, points_init, points_opt):
    title = "reconstruction_comparison"
    fig = plt.figure(title)
    ax = fig.add_subplot(111, projection="3d")
    display_interval = 20
    ax.scatter(
        points_gt_sdf[::display_interval, 0],
        points_gt_sdf[::display_interval, 1],
        points_gt_sdf[::display_interval, 2],
        color="green",
    )
    ax.scatter(
        points_init[::display_interval, 0],
        points_init[::display_interval, 1],
        points_init[::display_interval, 2],
        color="red",
    )
    ax.scatter(
        points_opt[::display_interval, 0],
        points_opt[::display_interval, 1],
        points_opt[::display_interval, 2],
        color="blue",
    )
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    min_coor = -0.08
    max_coor = 0.08
    ax.set_xlim(min_coor, max_coor)
    ax.set_ylim(min_coor, max_coor)
    ax.set_zlim(min_coor, max_coor)

    plt.show()
