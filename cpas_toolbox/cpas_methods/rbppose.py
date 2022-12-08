"""This module defines RBPPose interface.

Method is described in RBP-Pose: Residual Bounding Box Projection for Category-Level
Pose Estimation, Zhang, 2022

Implementation based on
[https://github.com/lolrudy/RBP_Pose](https://github.com/lolrudy/RBP_Pose).
"""
import copy
import os
from typing import TypedDict

import cv2
import numpy as np
import torch
import yoco
from scipy.spatial.transform import Rotation

from cpas_toolbox import camera_utils, quaternion_utils, utils
from cpas_toolbox.cpas_method import CPASMethod, PredictionDict

from . import _rbppose as rbppose


class RBPPose(CPASMethod):
    """Wrapper class for RBPPose."""

    class Config(TypedDict):
        """Configuration dictionary for RBPPose.

        Attributes:
            model: File path for model weights.
            model_url: URL to download model weights if file is not found.
            mean_shape: File path for mean shape file.
            mean_shape_url: URL to download mean shape file if it is not found.
            device: Device string for the model.
        """

        model: str
        model_url: str
        mean_shape: str
        mean_shape_url: str
        device: str

    default_config: Config = {
        "model": None,
        "model_url": None,
        "mean_shape": None,
        "mean_shape_url": None,
        "device": "cuda",
    }

    def __init__(self, config: Config, camera: camera_utils.Camera) -> None:
        """Initialize and load RBPPose model.

        Args:
            config: RBPPose configuration. See RBPPose.Config for more information.
            camera: Camera used for the input image.
        """
        config = yoco.load_config(config, current_dict=RBPPose.default_config)
        self._parse_config(config)
        self._camera = camera

    def _parse_config(self, config: Config) -> None:
        self._device = config["device"]
        self._model_file_path = utils.resolve_path(config["model"])
        self._model_url = config["model_url"]
        self._mean_shape_file_path = utils.resolve_path(config["mean_shape"])
        self._mean_shape_url = config["mean_shape_url"]

        # Check if files are available and download if not
        self._check_paths()

        # Initialize model
        self._net = rbppose.SSPN(...)
        self._net = self._net.to(self._device)
        state_dict = torch.load(self._model_file_path, map_location=self._device)
        cleaned_state_dict = copy.copy(state_dict)
        for key in state_dict.keys():
            if "face_recon" in key:
                cleaned_state_dict.pop(key)
            elif "pcl_encoder_prior" in key:
                cleaned_state_dict.pop(key)
        current_model_dict = self._net.state_dict()
        current_model_dict.update(cleaned_state_dict)
        self._net.load_state_dict(current_model_dict)
        self._net.eval()
        self._mean_shape_pointsets = np.load(self._mean_shape_file_path)

    def _check_paths(self) -> None:
        if not os.path.exists(self._model_file_path) or not os.path.exists(
            self._mean_shape_file_path
        ):
            print("RBPPose model weights not found, do you want to download to ")
            print("  ", self._model_file_path)
            print("  ", self._mean_shape_file_path)
            while True:
                decision = input("(Y/n) ").lower()
                if decision == "" or decision == "y":
                    self._download_weights()
                    break
                elif decision == "n":
                    print("RBPPose model weights not found. Aborting.")
                    exit(0)

    def _download_weights(self) -> None:
        if not os.path.exists(self._model_file_path):
            os.makedirs(os.path.dirname(self._model_file_path), exist_ok=True)
            utils.download(
                self._model_url,
                self._model_file_path,
            )
        if not os.path.exists(self._mean_shape_file_path):
            os.makedirs(os.path.dirname(self._mean_shape_file_path), exist_ok=True)
            utils.download(
                self._mean_shape_url,
                self._mean_shape_file_path,
            )

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_str: str,
    ) -> PredictionDict:
        """See cpas_toolbox.cpas_method.CPASMethod.inference.

        Based on https://github.com/lolrudy/RBP_Pose/blob/master/evaluation/evaluate.py
        """
        category_str_to_id = {
            "bottle": 0,
            "bowl": 1,
            "camera": 2,
            "can": 3,
            "laptop": 4,
            "mug": 5,
        }
        category_id = category_str_to_id[category_str]

        # Handle camera information
        fx, fy, cx, cy, _ = self._camera.get_pinhole_camera_parameters(0)
        width = self._camera.width
        height = self._camera.height
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        camera_matrix = torch.FloatTensor(camera_matrix).unsqueeze(0).to(self._device)

        # Prepare RGB crop (not used by default config)
        rgb_cv = color_image.numpy()[:, :, ::-1]  # BGR
        x1 = min(instance_mask.nonzero()[:, 1]).item()
        y1 = min(instance_mask.nonzero()[:, 0]).item()
        x2 = max(instance_mask.nonzero()[:, 1]).item()
        y2 = max(instance_mask.nonzero()[:, 0]).item()
        rmin, rmax, cmin, cmax = rbppose.get_bbox([y1, x1, y2, x2])
        cx = 0.5 * (cmin + cmax)
        cy = 0.5 * (rmin + rmax)
        bbox_center = np.array([cx, cy])  # (w/2, h/2)
        scale = min(max(cmax - cmin, rmax - rmin), max(height, width))
        rgb_crop = rbppose.crop_resize_by_warp_affine(
            rgb_cv,
            bbox_center,
            scale,
            rbppose.FLAGS.img_size,
            interpolation=cv2.INTER_NEAREST,
        ).transpose(2, 0, 1)
        rgb_crop = torch.FloatTensor(rgb_crop).unsqueeze(0).to(self._device)

        # Prepare depth crop (expected in mm)
        depth_cv = depth_image.numpy() * 1000
        depth_crop = rbppose.crop_resize_by_warp_affine(
            depth_cv,
            bbox_center,
            scale,
            rbppose.FLAGS.img_size,
            interpolation=cv2.INTER_NEAREST,
        )
        depth_crop = torch.FloatTensor(depth_crop)[None, None].to(self._device)

        # Prepare category
        category_input = torch.LongTensor([category_id]).to(self._device)

        # Prepare ROI Mask
        mask_np = instance_mask.float().numpy()
        roi_mask = rbppose.crop_resize_by_warp_affine(
            mask_np,
            bbox_center,
            scale,
            rbppose.FLAGS.img_size,
            interpolation=cv2.INTER_NEAREST,
        )
        roi_mask = torch.FloatTensor(roi_mask)[None, None].to(self._device)

        # Prepare mean shape (size?)
        mean_shape = rbppose.get_mean_shape(category_str) / 1000.0
        mean_shape = torch.FloatTensor(mean_shape).unsqueeze(0).to(self._device)

        # Prepare shape prior
        mean_shape_pointset = self._mean_shape_pointsets[category_id]
        shape_prior = (
            torch.FloatTensor(mean_shape_pointset).unsqueeze(0).to(self._device)
        )

        # Prepare 2D coordinates
        coord_2d = rbppose.get_2d_coord_np(width, height).transpose(1, 2, 0)
        roi_coord_2d = rbppose.crop_resize_by_warp_affine(
            coord_2d,
            bbox_center,
            scale,
            rbppose.FLAGS.img_size,
            interpolation=cv2.INTER_NEAREST,
        ).transpose(2, 0, 1)
        roi_coord_2d = torch.FloatTensor(roi_coord_2d).unsqueeze(0).to(self._device)

        output_dict = self._net(
            rgb=rgb_crop,
            depth=depth_crop,
            obj_id=category_input,
            camK=camera_matrix,
            def_mask=roi_mask,
            mean_shape=mean_shape,
            shape_prior=shape_prior,
            gt_2D=roi_coord_2d,
        )

        p_green_R_vec = output_dict["p_green_R"].detach().cpu()
        p_red_R_vec = output_dict["p_red_R"].detach().cpu()
        p_T = output_dict["Pred_T"].detach().cpu()
        f_green_R = output_dict["f_green_R"].detach().cpu()
        f_red_R = output_dict["f_red_R"].detach().cpu()
        sym = torch.FloatTensor(rbppose.get_sym_info(category_str)).unsqueeze(0)
        pred_RT = rbppose.generate_RT(
            [p_green_R_vec, p_red_R_vec],
            [f_green_R, f_red_R],
            p_T,
            mode="vec",
            sym=sym,
        )[0]
        position = output_dict["Pred_T"][0].detach().cpu()
        orientation_mat = pred_RT[:3, :3].detach().cpu()
        orientation = Rotation.from_matrix(orientation_mat.numpy())
        orientation_q = torch.FloatTensor(orientation.as_quat())
        extents = output_dict["Pred_s"][0].detach().cpu()
        scale = torch.linalg.norm(extents)
        reconstructed_points = output_dict["recon_model"][0].detach().cpu()
        reconstructed_points *= scale

        # Recenter for mug category
        if category_str == "mug":  # undo mug translation
            x_offset = (
                (
                    self._mean_shape_pointsets[5].max(axis=0)[0]
                    + self._mean_shape_pointsets[5].min(axis=0)[0]
                )
                / 2
                * scale
            )
            reconstructed_points[:, 0] -= x_offset
            position += quaternion_utils.quaternion_apply(
                orientation_q, torch.FloatTensor([x_offset, 0, 0])
            ).numpy()

        # NOCS Object -> ShapeNet Object convention
        obj_fix = torch.tensor([0.0, -1 / np.sqrt(2.0), 0.0, 1 / np.sqrt(2.0)])
        orientation_q = quaternion_utils.quaternion_multiply(orientation_q, obj_fix)
        reconstructed_points = quaternion_utils.quaternion_apply(
            quaternion_utils.quaternion_invert(obj_fix),
            reconstructed_points,
        )
        extents = torch.FloatTensor([extents[2], extents[1], extents[0]])

        return {
            "position": position,
            "orientation": orientation_q,
            "extents": extents,
            "reconstructed_pointcloud": reconstructed_points,
            "reconstructed_mesh": None,
        }
