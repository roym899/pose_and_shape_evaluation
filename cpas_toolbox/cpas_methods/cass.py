"""This module defines CASS interface.

Method is described in Learning Canonical Shape Space for Category-Level 6D Object Pose
and Size Estimation, Chen, 2020

Implementation based on https://github.com/densechen/CASS
"""
import os
from typing import TypedDict

import numpy as np
import torch
import torchvision.transforms.functional as TF
import yoco

from cpas_toolbox import camera_utils, pointset_utils, quaternion_utils, utils
from cpas_toolbox.cpas_method import CPASMethod, PredictionDict

from . import _cass as cass


class CASS(CPASMethod):
    """Wrapper class for CASS."""

    class Config(TypedDict):
        """Configuration dictionary for CASS.

        Attributes:
            model: Path to model.
            device: Device string for the model.
        """

        model: str

    default_config: Config = {
        "model": None,
    }

    def __init__(self, config: Config, camera: camera_utils.Camera) -> None:
        """Initialize and load CASS model.

        Args:
            config: CASS configuration. See CASS.Config for more information.
            camera: Camera used for the input image.
        """
        config = yoco.load_config(config, current_dict=CASS.default_config)
        self._parse_config(config)
        self._camera = camera

    def _parse_config(self, config: Config) -> None:
        self._device = config["device"]
        self._model_path = utils.resolve_path(config["model"])
        self._check_paths()
        self._cass = cass.CASS(
            num_points=config["num_points"], num_obj=config["num_objects"]
        )
        self._num_points = config["num_points"]
        self._cass.load_state_dict(
            torch.load(self._model_path, map_location=self._device)
        )
        self._cass.to(self._device)
        self._cass.eval()

    def _check_paths(self) -> None:
        if not os.path.exists(self._model_path):
            print("CASS model weights not found, do you want to download to ")
            print("  ", self._model_path)
            while True:
                decision = input("(Y/n) ").lower()
                if decision == "" or decision == "y":
                    self._download_weights()
                    break
                elif decision == "n":
                    print("CASS model weights not found. Aborting.")
                    exit(0)

    def _download_weights(self) -> None:
        if not os.path.exists(self._model_path):
            os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
            utils.download(
                "https://drive.google.com/u/0/uc?id=14K1a-Ft-YO9dUREEXxmWqF2ruUP4p7BZ&"
                "export=download",
                self._model_path,
            )

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_str: str,
    ) -> PredictionDict:
        """See CPASMethod.inference.

        Based on cass.tools.eval.
        """
        # get bounding box
        valid_mask = (depth_image != 0) * instance_mask
        rmin, rmax, cmin, cmax = cass.get_bbox(valid_mask.numpy())
        bb_mask = torch.zeros_like(depth_image)
        bb_mask[rmin:rmax, cmin:cmax] = 1.0

        # prepare image crop
        color_input = torch.flip(color_image, (2,)).permute([2, 0, 1])  # RGB -> BGR
        color_input = color_input[:, rmin:rmax, cmin:cmax]  # bb crop
        color_input = color_input.unsqueeze(0)  # add batch dim
        color_input = TF.normalize(
            color_input, mean=[0.51, 0.47, 0.44], std=[0.29, 0.27, 0.28]
        )

        # prepare points (fixed number of points, randomly picked)
        point_indices = valid_mask.nonzero()
        if len(point_indices) > self._num_points:
            subset = np.random.choice(
                len(point_indices), replace=False, size=self._num_points
            )
            point_indices = point_indices[subset]
        depth_mask = torch.zeros_like(depth_image)
        depth_mask[point_indices[:, 0], point_indices[:, 1]] = 1.0
        cropped_depth_mask = depth_mask[rmin:rmax, cmin:cmax]
        point_indices_input = cropped_depth_mask.flatten().nonzero()[:, 0]

        # prepare pointcloud
        points = pointset_utils.depth_to_pointcloud(
            depth_image,
            self._camera,
            normalize=False,
            mask=depth_mask,
            convention="opencv",
        )
        if len(points) < self._num_points:
            wrap_indices = np.pad(
                np.arange(len(points)), (0, self._num_points - len(points)), mode="wrap"
            )
            points = points[wrap_indices]
            point_indices_input = point_indices_input[wrap_indices]

        # x, y inverted for some reason...
        points[:, 0] *= -1
        points[:, 1] *= -1
        points = points.unsqueeze(0)
        point_indices_input = point_indices_input.unsqueeze(0)

        # move inputs to device
        color_input = color_input.to(self._device)
        points = points.to(self._device)
        point_indices_input = point_indices_input.to(self._device)

        category_str_to_id = {
            "bottle": 0,
            "bowl": 1,
            "camera": 2,
            "can": 3,
            "laptop": 4,
            "mug": 5,
        }
        category_id = category_str_to_id[category_str]

        # CASS model uses 0-indexed categories, same order as NOCSDataset
        category_index = torch.tensor([category_id], device=self._device)

        # Call CASS network
        folding_encode = self._cass.foldingnet.encode(
            color_input, points, point_indices_input
        )
        posenet_encode = self._cass.estimator.encode(
            color_input, points, point_indices_input
        )
        pred_r, pred_t, pred_c = self._cass.estimator.pose(
            torch.cat([posenet_encode, folding_encode], dim=1), category_index
        )
        reconstructed_points = self._cass.foldingnet.recon(folding_encode)[0]

        # Postprocess outputs
        reconstructed_points = reconstructed_points.view(-1, 3).cpu()
        pred_c = pred_c.view(1, self._num_points)
        _, max_index = torch.max(pred_c, 1)
        pred_t = pred_t.view(self._num_points, 1, 3)
        orientation_q = pred_r[0][max_index[0]].view(-1).cpu()
        points = points.view(self._num_points, 1, 3)
        position = (points + pred_t)[max_index[0]].view(-1).cpu()
        # output is scalar-first -> scalar-last
        orientation_q = torch.tensor([*orientation_q[1:], orientation_q[0]])

        # Flip x and y axis of position and orientation (undo flipping of points)
        # (x-left, y-up, z-forward) convention -> OpenCV convention
        position[0] *= -1
        position[1] *= -1
        cam_fix = torch.tensor([0.0, 0.0, 1.0, 0.0])
        # NOCS Object -> ShapeNet Object convention
        obj_fix = torch.tensor(
            [0.0, -1 / np.sqrt(2.0), 0.0, 1 / np.sqrt(2.0)]
        )  # CASS object to ShapeNet object
        orientation_q = quaternion_utils.quaternion_multiply(cam_fix, orientation_q)
        orientation_q = quaternion_utils.quaternion_multiply(orientation_q, obj_fix)
        reconstructed_points = quaternion_utils.quaternion_apply(
            quaternion_utils.quaternion_invert(obj_fix),
            reconstructed_points,
        )

        # TODO refinement code from cass.tools.eval? (not mentioned in paper??)

        extents, _ = reconstructed_points.abs().max(dim=0)
        extents *= 2.0

        # pointset_utils.visualize_pointset(reconstructed_points)
        return {
            "position": position.detach(),
            "orientation": orientation_q.detach(),
            "extents": extents.detach(),
            "reconstructed_pointcloud": reconstructed_points.detach(),
            "reconstructed_mesh": None,
        }
