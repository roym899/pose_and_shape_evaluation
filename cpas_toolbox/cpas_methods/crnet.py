"""This module defines CR-Net interface.

Method is described in Category-Level 6D Object Pose Estimation via Cascaded Relation
and Recurrent Reconstruction Networks, Wang, 2021.

Implementation based on
https://github.com/JeremyWANGJZ/Category-6D-Pose
"""
import os
from typing import TypedDict

import numpy as np
import torch
import yoco

from cpas_toolbox import camera_utils, utils
from cpas_toolbox.cpas_method import CPASMethod

from . import _crnet as crnet


class CRNet(CPASMethod):
    """Wrapper class for CRNet."""

    class Config(TypedDict):
        """Configuration dictionary for CRNet.

        Attributes:
            model: Path to model.
            num_categories: Number of categories used by model.
            num_shape_points: Number of points in shape prior.
            device: Device string for the model.
        """

        model: str
        num_categories: int

    default_config: Config = {
        "model": None,
        "num_categories": None,
        "num_shape_points": None,
    }

    def __init__(self, config: Config, camera: camera_utils.Camera) -> None:
        """Initialize and load CRNet model.

        Args:
            config: CRNet configuration. See CRNet.Config for more information.
            camera: Camera used for the input image.
        """
        config = yoco.load_config(config, current_dict=CRNet.default_config)
        self._parse_config(config)
        self._camera = camera

    def _parse_config(self, config: Config) -> None:
        self._device = config["device"]
        self._model_path = utils.resolve_path(config["model"])
        self._model_url = config["model_url"]
        self._mean_shape_path = utils.resolve_path(config["mean_shape"])
        self._mean_shape_url = config["mean_shape_url"]
        self._check_paths()
        self._crnet = crnet.DeformNet(
            config["num_categories"], config["num_shape_points"]
        )
        self._crnet.cuda()
        self._crnet = torch.nn.DataParallel(self._crnet, device_ids=[self._device])
        self._crnet.load_state_dict(torch.load(self._model_path, map_location="cuda"))
        self._crnet.eval()
        self._mean_shape_pointsets = np.load(self._mean_shape_path)
        self._num_input_points = config["num_input_points"]
        self._image_size = config["image_size"]

    def _check_paths(self) -> None:
        if not os.path.exists(self._model_path) or not os.path.exists(
            self._mean_shape_path
        ):
            print("CRNet model weights not found, do you want to download to ")
            print("  ", self._model_path)
            print("  ", self._mean_shape_path)
            while True:
                decision = input("(Y/n) ").lower()
                if decision == "" or decision == "y":
                    self._download_weights()
                    break
                elif decision == "n":
                    print("CRNet model weights not found. Aborting.")
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
