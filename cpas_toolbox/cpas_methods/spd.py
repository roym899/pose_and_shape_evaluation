"""This module defines SPD interface.

Method is described in Shape Prior Deformation for Categorical 6D Object Pose and Size
Estimation, Tian, 2020.

Implementation based on
https://github.com/mentian/object-deformnet
"""
import os
import shutil
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

from . import _spd as spd


class SPD(CPASMethod):
    """Wrapper class for Shape Prior Deformation (SPD)."""

    class Config(TypedDict):
        """Configuration dictionary for SPD.

        Attributes:
            model_file: Path to model.
            mean_shape_file: Path to mean shape file.
            num_categories: Number of categories used by model.
            num_shape_points: Number of points in shape prior.
            num_input_points: Number of input points.
            image_size: Size of image crop.
            device: Device string for the model.
        """

        model_file: str
        mean_shape_file: str
        num_categories: int
        num_shape_points: int
        num_input_points: int
        image_size: int
        device: str

    default_config: Config = {
        "model_file": None,
        "mean_shape_file": None,
        "num_categories": None,
        "num_shape_points": None,
        "num_input_points": None,
        "image_size": None,
        "device": "cuda",
    }

    def __init__(self, config: Config, camera: camera_utils.Camera) -> None:
        """Initialize and load SPD model.

        Args:
            config: SPD configuration. See SPD.Config for more information.
            camera: Camera used for the input image.
        """
        config = yoco.load_config(config, current_dict=SPD.default_config)
        self._parse_config(config)
        self._camera = camera

    def _parse_config(self, config: Config) -> None:
        self._device = config["device"]
        self._model_file_path = utils.resolve_path(config["model_file"])
        self._mean_shape_file_path = utils.resolve_path(config["mean_shape_file"])
        self._check_paths()
        self._spd_net = spd.DeformNet(
            config["num_categories"], config["num_shape_points"]
        )
        self._spd_net.to(self._device)
        self._spd_net.load_state_dict(
            torch.load(self._model_file_path, map_location=self._device)
        )
        self._spd_net.eval()
        self._mean_shape_pointsets = np.load(self._mean_shape_file_path)
        self._num_input_points = config["num_input_points"]
        self._image_size = config["image_size"]

    def _check_paths(self) -> None:
        if not os.path.exists(self._model_file_path) or not os.path.exists(
            self._mean_shape_file_path
        ):
            print("SPD model weights not found, do you want to download to ")
            print("  ", self._model_file_path)
            print("  ", self._mean_shape_file_path)
            while True:
                decision = input("(Y/n) ").lower()
                if decision == "" or decision == "y":
                    self._download_weights()
                    break
                elif decision == "n":
                    print("SPD model weights not found. Aborting.")
                    exit(0)

    def _download_weights(self) -> None:
        if not os.path.exists(self._model_file_path):
            os.makedirs(os.path.dirname(self._model_file_path), exist_ok=True)
            download_dir_path = os.path.dirname(self._model_file_path)
            zip_path = os.path.join(download_dir_path, "temp.zip")
            utils.download(
                "https://drive.google.com/u/0/uc?id=1p72NdY4Bie_sra9U8zoUNI4fTrQZdbnc&"
                "export=download",
                zip_path,
            )
            z = zipfile.ZipFile(zip_path)
            z.extract("deformnet_eval/real/model_50.pth", download_dir_path)
            z.close()
            os.remove(zip_path)
            shutil.move(
                os.path.join(
                    download_dir_path, "deformnet_eval", "real", "model_50.pth"
                ),
                download_dir_path,
            )
            shutil.rmtree(os.path.join(download_dir_path, "deformnet_eval"))
        if not os.path.exists(self._mean_shape_file_path):
            os.makedirs(os.path.dirname(self._mean_shape_file_path), exist_ok=True)
            utils.download(
                "https://github.com/mentian/object-deformnet/raw/master/assets/"
                "mean_points_emb.npy",
                self._mean_shape_file_path,
            )

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_str: str,
    ) -> PredictionDict:
        """See MethodWrapper.inference.

        Based on spd.evaluate.
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

        # get bounding box
        x1 = min(instance_mask.nonzero()[:, 1]).item()
        y1 = min(instance_mask.nonzero()[:, 0]).item()
        x2 = max(instance_mask.nonzero()[:, 1]).item()
        y2 = max(instance_mask.nonzero()[:, 0]).item()
        rmin, rmax, cmin, cmax = spd.get_bbox([y1, x1, y2, x2])
        bb_mask = torch.zeros_like(depth_image)
        bb_mask[rmin:rmax, cmin:cmax] = 1.0

        valid_mask = (depth_image != 0) * instance_mask

        # prepare image crop
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
        color_input = color_input.unsqueeze(0)  # add batch dim

        # convert depth to pointcloud
        fx, fy, cx, cy, _ = self._camera.get_pinhole_camera_parameters(pixel_center=0.0)
        width = self._camera.width
        height = self._camera.height
        point_indices = valid_mask[rmin:rmax, cmin:cmax].numpy().flatten().nonzero()[0]
        xmap = np.array([[i for i in range(width)] for _ in range(height)])
        ymap = np.array([[j for _ in range(width)] for j in range(height)])
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

        # move inputs to device
        color_input = color_input.to(self._device)
        points = torch.Tensor(points).unsqueeze(0).to(self._device)
        point_indices = torch.LongTensor(point_indices).unsqueeze(0).to(self._device)
        category_id = torch.LongTensor([category_id]).to(self._device)
        mean_shape_pointset = (
            torch.Tensor(mean_shape_pointset).unsqueeze(0).to(self._device)
        )

        # Call SPD network
        assign_matrix, deltas = self._spd_net(
            points, color_input, point_indices, category_id, mean_shape_pointset
        )

        # Postprocess outputs
        inst_shape = mean_shape_pointset + deltas
        assign_matrix = torch.softmax(assign_matrix, dim=2)
        coords = torch.bmm(assign_matrix, inst_shape)  # (1, n_pts, 3)

        point_indices = point_indices[0].cpu().numpy()
        _, point_indices = np.unique(point_indices, return_index=True)
        nocs_coords = coords[0, point_indices, :].detach().cpu().numpy()
        extents = 2 * np.amax(np.abs(inst_shape[0].detach().cpu().numpy()), axis=0)
        points = points[0, point_indices, :].cpu().numpy()
        scale, orientation_m, position, _ = spd.align.estimateSimilarityTransform(
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
