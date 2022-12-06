"""This module defines DPDN interface.

Method is described in * Category-Level 6D Object Pose and Size Estimation using
Self-Supervised Deep Prior Deformation Networks, Lin, 2022.

Implementation based on
https://github.com/JiehongLin/Self-DPDN
"""
import os
import shutil
import tempfile
import zipfile
from typing import TypedDict

import numpy as np
import torch
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
        "image_size": None,
        "model": None,
        "model_url": None,
        "mean_shape": None,
        "mean_shape_url": None,
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

        self._dpdn = dpdn.Net(config["num_categories"], config["num_shape_points"])
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
        exit()

        # Call DPDN
        assign_matrix, deltas = self._dpdn(
            points,
            color_input,
            point_indices,
            category_id,
            mean_shape_pointset,
        )
