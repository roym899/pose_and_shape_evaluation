"""This module defines CR-Net interface.

Method is described in SGPA: Structure-Guided Prior Adaptation for Category-Level 6D
Object Pose Estimation, Chen, 2021.

Implementation based on https://github.com/ck-kai/SGPA
"""
import os
from typing import TypedDict

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yoco
from scipy.spatial.transform import Rotation

from cpas_toolbox import camera_utils, quaternion_utils, utils
from cpas_toolbox.cpas_method import CPASMethod, PredictionDict

from . import _sgpa as sgpa


class SGPA(CPASMethod):
    """Wrapper class for SGPA."""

    class Config(TypedDict):
        """Configuration dictionary for SGPA.

        Attributes:
            model: Path to model.
            num_categories: Number of categories used by model.
            num_shape_points: Number of points in shape prior.
            num_input_points: Number of 3D input points.
            num_structure_points: Number of keypoints used for pose estimation.
            image_size: Image size consumed by network (crop will be resized to this).
            model: File path for model weights.
            model_url: URL to download model weights if file is not found.
            mean_shape: File path for mean shape file.
            mean_shape_url: URL to download mean shape file if it is not found.
            device: Device string for the model.
        """

        model: str
        num_categories: int
        num_shape_points: int
        num_input_points: int
        num_structure_points: int
        image_size: int
        model: str
        model_url: str
        mean_shape: str
        mean_shape_url: str
        device: str

    default_config: Config = {
        "model": None,
        "num_categories": None,
        "num_shape_points": None,
        "num_input_points": None,
        "num_structure_points": None,
        "image_size": None,
        "model": None,
        "model_url": None,
        "mean_shape": None,
        "mean_shape_url": None,
        "device": "cuda",
    }

    def __init__(self, config: Config, camera: camera_utils.Camera) -> None:
        """Initialize and load SGPA model.

        Args:
            config: SGPA configuration. See SGPA.Config for more information.
            camera: Camera used for the input image.
        """
        config = yoco.load_config(config, current_dict=SGPA.default_config)
        self._parse_config(config)
        self._camera = camera

    def _parse_config(self, config: Config) -> None:
        self._device = config["device"]
        self._model_path = utils.resolve_path(config["model"])
        self._model_url = config["model_url"]
        self._mean_shape_path = utils.resolve_path(config["mean_shape"])
        self._mean_shape_url = config["mean_shape_url"]
        self._check_paths()
        self._sgpa = sgpa.SGPANet(
            config["num_categories"],
            config["num_shape_points"],
            num_structure_points=config["num_structure_points"],
        )
        self._sgpa.cuda()
        self._sgpa = torch.nn.DataParallel(self._sgpa, device_ids=[self._device])
        self._sgpa.load_state_dict(
            torch.load(self._model_path, map_location=self._device)
        )
        self._sgpa.eval()
        self._mean_shape_pointsets = np.load(self._mean_shape_path)
        self._num_input_points = config["num_input_points"]
        self._image_size = config["image_size"]

    def _check_paths(self) -> None:
        if not os.path.exists(self._model_path) or not os.path.exists(
            self._mean_shape_path
        ):
            print("SGPA model weights not found, do you want to download to ")
            print("  ", self._model_path)
            print("  ", self._mean_shape_path)
            while True:
                decision = input("(Y/n) ").lower()
                if decision == "" or decision == "y":
                    self._download_weights()
                    break
                elif decision == "n":
                    print("SGPA model weights not found. Aborting.")
                    exit(0)

    def _download_weights(self) -> None:
        if not os.path.exists(self._model_path):
            os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
            utils.download(
                self._model_url,
                self._model_path,
            )
        if not os.path.exists(self._mean_shape_path):
            os.makedirs(os.path.dirname(self._mean_shape_path), exist_ok=True)
            utils.download(
                self._mean_shape_url,
                self._mean_shape_path,
            )

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_str: str,
    ) -> PredictionDict:
        """See cpas_toolbox.cpas_method.CPASMethod.inference.

        Based on https://github.com/JeremyWANGJZ/Category-6D-Pose/blob/main/evaluate.py
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

        # Get bounding box
        x1 = min(instance_mask.nonzero()[:, 1]).item()
        y1 = min(instance_mask.nonzero()[:, 0]).item()
        x2 = max(instance_mask.nonzero()[:, 1]).item()
        y2 = max(instance_mask.nonzero()[:, 0]).item()
        rmin, rmax, cmin, cmax = sgpa.get_bbox([y1, x1, y2, x2])

        valid_mask = (depth_image != 0) * instance_mask

        # Prepare image crop
        color_input = color_image[rmin:rmax, cmin:cmax, :].numpy()  # bb crop
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

        # Prepare input points
        fx, fy, cx, cy, _ = self._camera.get_pinhole_camera_parameters(pixel_center=0.0)
        width = self._camera.width
        height = self._camera.height
        point_indices = valid_mask[rmin:rmax, cmin:cmax].numpy().flatten().nonzero()[0]
        xmap = np.array([[i for i in range(width)] for _ in range(height)])
        ymap = np.array([[j for _ in range(width)] for j in range(height)])
        # if len(choose) < 32:
        #     f_sRT[i] = np.identity(4, dtype=float)
        #     f_size[i] = 2 * np.amax(np.abs(mean_shape_pointset), axis=0)
        #     continue
        # else:
        #     valid_inst.append(i)
        if len(point_indices) > self._num_input_points:
            # take subset of points if two many depth points
            point_indices_mask = np.zeros(len(point_indices), dtype=int)
            point_indices_mask[: self._num_input_points] = 1
            np.random.shuffle(point_indices_mask)
            point_indices = point_indices[point_indices_mask.nonzero()]
        else:
            point_indices = np.pad(
                point_indices, (0, self._num_input_points - len(point_indices)), "wrap"
            )  # repeat points if not enough depth observation
        depth_masked = depth_image[rmin:rmax, cmin:cmax].flatten()[point_indices][
            :, None
        ]
        xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[point_indices][:, None]
        ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[point_indices][:, None]
        pt2 = depth_masked.numpy()
        pt0 = (xmap_masked - cx) * pt2 / fx
        pt1 = (ymap_masked - cy) * pt2 / fy
        points = np.concatenate((pt0, pt1, pt2), axis=1)
        # adjust indices for resizing of color image
        crop_w = rmax - rmin
        ratio = self._image_size / crop_w
        col_idx = point_indices % crop_w
        row_idx = point_indices // crop_w
        point_indices = (
            np.floor(row_idx * ratio) * self._image_size + np.floor(col_idx * ratio)
        ).astype(np.int64)

        # Move inputs to device and convert to right shape
        color_input = color_input.unsqueeze(0).to(self._device)
        points = torch.cuda.FloatTensor(points).unsqueeze(0).to(self._device)
        point_indices = torch.LongTensor(point_indices).unsqueeze(0).to(self._device)
        category_id = torch.cuda.LongTensor([category_id]).to(self._device)
        mean_shape_pointset = (
            torch.FloatTensor(mean_shape_pointset).unsqueeze(0).to(self._device)
        )

        # Call SGPA
        _, assign_matrix, deltas = self._sgpa(
            points,
            color_input,
            point_indices,
            category_id,
            mean_shape_pointset,
        )

        # Postprocess output
        inst_shape = mean_shape_pointset + deltas
        assign_matrix = torch.softmax(assign_matrix, dim=2)
        coords = torch.bmm(assign_matrix, inst_shape)  # (1, n_pts, 3)

        point_indices = point_indices[0].cpu().numpy()
        _, point_indices = np.unique(point_indices, return_index=True)
        nocs_coords = coords[0, point_indices, :].detach().cpu().numpy()
        extents = 2 * np.amax(np.abs(inst_shape[0].detach().cpu().numpy()), axis=0)
        points = points[0, point_indices, :].cpu().numpy()
        scale, orientation_m, position, _ = sgpa.estimateSimilarityTransform(
            nocs_coords, points
        )
        orientation_q = torch.Tensor(Rotation.from_matrix(orientation_m).as_quat())

        reconstructed_points = inst_shape[0].detach().cpu() * scale

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
        extents, _ = reconstructed_points.abs().max(dim=0)
        extents *= 2.0

        return {
            "position": torch.Tensor(position),
            "orientation": orientation_q,
            "extents": torch.Tensor(extents),
            "reconstructed_pointcloud": reconstructed_points,
            "reconstructed_mesh": None,
        }
