"""This module defines SDFEst interface.

Method is described in SDFEst: Categorical Pose and Shape Estimation of Objects From
RGB-D Using Signed Distance Fields, Bruns, 2022.


Implementation based on
https://github.com/roym899/sdfest/
"""
from typing import TypedDict

import numpy as np
import sdfest
import torch
import yoco
from sdfest.estimation.simple_setup import SDFPipeline

from cpas_toolbox import camera_utils, pointset_utils
from cpas_toolbox.cpas_method import CPASMethod, PredictionDict


class SDFEst(CPASMethod):
    """Wrapper class for SDFEst."""

    class Config(TypedDict):
        """Configuration dictionary for SDFEst.

        All keys supported by SDFPipeline are supported and will overwrite config
        contained in sdfest_... files. The keys specified here are used by this
        script only.

        The two keys sdfest_..._config_files  will be parsed with SDFEst install
        directory as part of the search paths. This allows to use the default config
        that comes with SDFEst installation.

        Attributes:
            sdfest_default_config_file: Default configuration file loaded first.
            sdfest_category_config_files: Per-category configuration file loaded second.
            device: Device used for computation.
            num_points: Numbner of points extracted from mesh.
            prior: Prior distribution to modify orientation distribution.
            visualize_optimization:
                Whether to show additional optimization visualization.
        """

    default_config: Config = {
        "sdfest_default_config_file": "estimation/configs/default.yaml",
        "sdfest_category_config_files": {
            "bottle": "estimation/configs/models/bottle.yaml",
            "bowl": "estimation/configs/models/bowl.yaml",
            "laptop": "estimation/configs/models/laptop.yaml",
            "can": "estimation/configs/models/can.yaml",
            "camera": "estimation/configs/models/camera.yaml",
            "mug": "estimation/configs/models/mug.yaml",
        },
    }

    def __init__(self, config: Config, camera: camera_utils.Camera) -> None:
        """Initialize and load SDFEst models.

        Configuration loaded in following order
            sdfest_default_config_file -> sdfest_category_config_files -> all other keys
        I.e., keys specified directly will take precedence over keys specified in
        default file.
        """
        default_config = yoco.load_config_from_file(
            config["sdfest_default_config_file"],
            search_paths=[".", "~/.sdfest/", sdfest.__path__[0]],
        )

        self._pipeline_dict = {}  # maps category to category-specific pipeline
        self._device = config["device"]
        self._visualize_optimization = config["visualize_optimization"]
        self._num_points = config["num_points"]
        self._prior = config["prior"] if "prior" in config else None

        # create per-categry models
        for category_str in config["sdfest_category_config_files"].keys():
            category_config = yoco.load_config_from_file(
                config["sdfest_category_config_files"][category_str],
                current_dict=default_config,
                search_paths=[".", "~/.sdfest/", sdfest.__path__[0]],
            )
            category_config = yoco.load_config(config, category_config)
            self._pipeline_dict[category_str] = SDFPipeline(category_config)
            self._pipeline_dict[category_str].cam = camera

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_str: str,
    ) -> PredictionDict:
        """See CPASMethod.inference."""
        # skip unsupported category
        if category_str not in self._pipeline_dict:
            return {
                "position": torch.tensor([0, 0, 0]),
                "orientation": torch.tensor([0, 0, 0, 1]),
                "extents": torch.tensor([1, 1, 1]),
                "reconstructed_pointcloud": torch.tensor([[0, 0, 0]]),
                "reconstructed_mesh": None,
            }

        pipeline = self._pipeline_dict[category_str]

        # move inputs to device
        color_image = color_image.to(self._device)
        depth_image = depth_image.to(self._device, copy=True)
        instance_mask = instance_mask.to(self._device)

        if self._prior is not None:
            prior = torch.tensor(self._prior[category_str], device=self._device)
            prior /= torch.sum(prior)
        else:
            prior = None

        position, orientation, scale, shape = pipeline(
            depth_image,
            instance_mask,
            color_image,
            visualize=self._visualize_optimization,
            prior_orientation_distribution=prior,
        )

        # outputs of SDFEst are OpenGL camera, ShapeNet object convention
        position_cv = pointset_utils.change_position_camera_convention(
            position[0], "opengl", "opencv"
        )
        orientation_cv = pointset_utils.change_orientation_camera_convention(
            orientation[0], "opengl", "opencv"
        )

        # reconstruction + extent
        mesh = pipeline.generate_mesh(shape, scale, True).get_transformed_o3d_geometry()
        reconstructed_points = torch.from_numpy(
            np.asarray(mesh.sample_points_uniformly(self._num_points).points)
        )
        extents, _ = reconstructed_points.abs().max(dim=0)
        extents *= 2.0

        return {
            "position": position_cv.detach().cpu(),
            "orientation": orientation_cv.detach().cpu(),
            "extents": extents,
            "reconstructed_pointcloud": reconstructed_points,
            "reconstructed_mesh": mesh,
        }
