"""This module defines RBPPose interface.

Method is described in RBP-Pose: Residual Bounding Box Projection for Category-Level
Pose Estimation, Zhang, 2022

Implementation based on
[https://github.com/lolrudy/RBP_Pose](https://github.com/lolrudy/RBP_Pose).
"""
import copy
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
        exit()
