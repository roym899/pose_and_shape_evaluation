"""This module defines DPDN interface.

Method is described in Category-Level 6D Object Pose and Size Estimation using
Self-Supervised Deep Prior Deformation Networks, Lin, 2022.

Implementation based on
[https://github.com/JiehongLin/Self-DPDN](https://github.com/JiehongLin/Self-DPDN).
"""
import os
import shutil
import tempfile
import zipfile
from typing import TypedDict

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yoco
from scipy.spatial.transform import Rotation

from cpas_toolbox import camera_utils, quaternion_utils, utils
from cpas_toolbox.cpas_method import CPASMethod, PredictionDict

from . import _dpdn as dpdn


class DPDN(CPASMethod):
    """Wrapper class for DPDN."""

    class Config(TypedDict):
        """Configuration dictionary for DPDN.

        Attributes:
            model: Path to model.
            num_categories: Number of categories used by model.
            num_shape_points: Number of points in shape prior.
            device: Device string for the model.
        """

        model: str
        num_categories: int
        num_shape_points: int
        num_input_points: int
        image_size: int
        model: str
        model_url: str
        mean_shape: str
        mean_shape_url: str
        resnet_dir: str
        device: str

    default_config: Config = {
        "model": None,
        "num_categories": None,
        "num_shape_points": None,
        "num_input_points": None,
        "image_size": None,
        "model": None,
        "model_url": None,
        "mean_shape": None,
        "mean_shape_url": None,
        "resnet_dir": None,
        "device": "cuda",
    }

    def __init__(self, config: Config, camera: camera_utils.Camera) -> None:
        """Initialize and load DPDN model.

        Args:
            config: DPDN configuration. See DPDN.Config for more information.
            camera: Camera used for the input image.
        """
        config = yoco.load_config(config, current_dict=DPDN.default_config)
        self._parse_config(config)
        self._camera = camera

    def _parse_config(self, config: Config) -> None:
        self._device = config["device"]
        self._model_file_path = utils.resolve_path(config["model"])
        self._model_url = config["model_url"]
        self._mean_shape_file_path = utils.resolve_path(config["mean_shape"])
        self._mean_shape_url = config["mean_shape_url"]
        self._check_paths()
        self._resnet_dir_path = utils.resolve_path(config["resnet_dir"])

        self._dpdn = dpdn.Net(
            config["num_categories"], config["num_shape_points"], self._resnet_dir_path
        )
        self._dpdn = self._dpdn.to(self._device)
        checkpoint = torch.load(self._model_file_path, map_location=self._device)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict":
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self._dpdn.load_state_dict(state_dict)
        self._dpdn.eval()
        self._mean_shape_pointsets = np.load(self._mean_shape_file_path)
        self._num_input_points = config["num_input_points"]
        self._image_size = config["image_size"]

    def _check_paths(self) -> None:
        if not os.path.exists(self._model_file_path) or not os.path.exists(
            self._mean_shape_file_path
        ):
            print("DPDN model weights not found, do you want to download to ")
            print("  ", self._model_file_path)
            print("  ", self._mean_shape_file_path)
            while True:
                decision = input("(Y/n) ").lower()
                if decision == "" or decision == "y":
                    self._download_weights()
                    break
                elif decision == "n":
                    print("DPDN model weights not found. Aborting.")
                    exit(0)

    def _download_weights(self) -> None:
        if not os.path.exists(self._model_file_path):
            dl_dir_path = tempfile.mkdtemp()
            print(dl_dir_path)
            zip_file_path = os.path.join(dl_dir_path, "temp")
            os.makedirs(os.path.dirname(self._model_file_path), exist_ok=True)
            utils.download(
                self._model_url,
                zip_file_path,
            )
            z = zipfile.ZipFile(zip_file_path)
            z.extract("log/supervised/epoch_30.pth", dl_dir_path)
            z.close()
            os.remove(zip_file_path)
            shutil.move(
                os.path.join(dl_dir_path, "log", "supervised", "epoch_30.pth"),
                self._model_file_path,
            )
            shutil.rmtree(dl_dir_path)
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

        Based on https://github.com/JiehongLin/Self-DPDN/blob/main/test.py
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
        mean_shape_pointset = self._mean_shape_pointsets[category_id]

        input_dict = {}

        # Get bounding box
        x1 = min(instance_mask.nonzero()[:, 1]).item()
        y1 = min(instance_mask.nonzero()[:, 0]).item()
        x2 = max(instance_mask.nonzero()[:, 1]).item()
        y2 = max(instance_mask.nonzero()[:, 0]).item()
        rmin, rmax, cmin, cmax = dpdn.get_bbox([y1, x1, y2, x2])

        # Prepare image crop
        color_input = color_image[rmin:rmax, cmin:cmax, :].numpy()
        color_input = cv2.resize(
            color_input,
            (self._image_size, self._image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        color_input = TF.normalize(
            TF.to_tensor(color_input),  # (H, W, C) -> (C, H, W), RGB
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        input_dict["rgb"] = color_input.unsqueeze(0).to(self._device)

        # Prepare point indices
        mask = (depth_image != 0) * instance_mask
        cropped_mask = mask[rmin:rmax, cmin:cmax]
        cropped_mask_indices = cropped_mask.numpy().flatten().nonzero()[0]
        if len(cropped_mask_indices) <= self._num_input_points:
            indices = np.random.choice(
                len(cropped_mask_indices), self._num_input_points
            )
        else:
            indices = np.random.choice(
                len(cropped_mask_indices), self._num_input_points, replace=False
            )
        chosen_cropped_indices = cropped_mask_indices[indices]

        # adjust indices for resizing of color image
        crop_w = rmax - rmin
        ratio = self._image_size / crop_w
        col_idx = chosen_cropped_indices % crop_w
        row_idx = chosen_cropped_indices // crop_w
        final_cropped_indices = (
            np.floor(row_idx * ratio) * self._image_size + np.floor(col_idx * ratio)
        ).astype(np.int64)
        input_dict["choose"] = (
            torch.LongTensor(final_cropped_indices).unsqueeze(0).to(self._device)
        )

        # Prepare input points
        fx, fy, cx, cy, _ = self._camera.get_pinhole_camera_parameters(pixel_center=0.0)
        width = self._camera.width
        height = self._camera.height
        depth_image_np = depth_image.numpy()
        depth_image_np = dpdn.fill_missing(depth_image_np * 1000.0, 1000.0, 1) / 1000.0

        xmap = np.array([[i for i in range(width)] for _ in range(height)])
        ymap = np.array([[j for _ in range(width)] for j in range(height)])
        pts2 = depth_image_np.copy()
        pts0 = (xmap - cx) * pts2 / fx
        pts1 = (ymap - cy) * pts2 / fy
        pts_map = np.stack([pts0, pts1, pts2])
        pts_map = np.transpose(pts_map, (1, 2, 0)).astype(np.float32)
        cropped_pts_map = pts_map[rmin:rmax, cmin:cmax, :]
        input_points = cropped_pts_map.reshape((-1, 3))[chosen_cropped_indices, :]
        input_dict["pts"] = (
            torch.FloatTensor(input_points).unsqueeze(0).to(self._device)
        )

        # Prepare prior
        input_dict["prior"] = (
            torch.FloatTensor(mean_shape_pointset).unsqueeze(0).to(self._device)
        )

        # Prepare category id
        input_dict["category_label"] = torch.cuda.LongTensor([category_id]).to(
            self._device
        )

        # Call DPDN
        outputs = self._dpdn(input_dict)

        # Convert outputs to expected format
        position = outputs["pred_translation"][0].detach().cpu()

        orientation_mat = outputs["pred_rotation"][0].detach().cpu()
        orientation = Rotation.from_matrix(orientation_mat.detach().numpy())
        orientation_q = torch.FloatTensor(orientation.as_quat())
        extents = outputs["pred_size"][0].detach().cpu()
        reconstructed_points = outputs["pred_qv"][0].detach().cpu()
        scale = torch.linalg.norm(extents)
        reconstructed_points *= scale

        # Recenter for mug category
        # TODO not really sure if this is correct, but seems to give best results
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
