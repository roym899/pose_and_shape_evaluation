"""This module defines ASMNet interface.

Method is described in ASM-Net: Category-level Pose and Shape, Akizuki, 2021

Implementation based on https://github.com/sakizuki/asm-net
"""
import copy
import os
import shutil
import tempfile
import zipfile
from typing import List, TypedDict

import numpy as np
import open3d as o3d
import torch
import yoco
from scipy.spatial.transform import Rotation

from cpas_toolbox import camera_utils, quaternion_utils, utils
from cpas_toolbox.cpas_method import CPASMethod, PredictionDict

from . import _asmnet as asmnet


class ASMNet(CPASMethod):
    """Wrapper class for ASMNet."""

    class Config(TypedDict):
        """Configuration dictionary for ASMNet.

        Attributes:
            model: Path to model.
            device: Device string for the model.
            models_dir:
                Path to directory containing model parameters.
                Must contain the following directory structure:
                    {models_dir}/{category_0}/model.pth
                    ...
            asm_params_dir:
                Path to direactory containing ASM parameters.
                Must contain the following directory structure:
                    {asm_params_directory}/{category_0}/train/info.npz
                    ...
            weights_url:
                URL to download model and ASM params from if they do not exist yet.
            categories:
                List of categories. Each category requires corresponding directory with
                model.pth and info.npz. See models_dir and asm_params_dir.
            num_points: Number of input points.
            deformation_dimension: Number of deformation parameters.
            use_mean_shape:
                Whether the mean shape (0) or the predicted shape deformation should
                be used.
            use_icp: Whether to use ICP to refine the pose.
        """

        models_dir: str
        asm_params_dir: str
        weights_url: str
        device: str
        categories: List[str]
        num_points: int
        deformation_dimension: int
        use_mean_shape: bool
        use_icp: bool

    default_config: Config = {
        "model_params_dir": None,
        "asm_params_dir": None,
        "weights_url": None,
        "device": "cuda",
        "categories": [],
        "num_points": 800,
        "deformation_dimension": 3,
        "use_mean_shape": False,
        "use_icp": True,
    }

    def __init__(self, config: Config, camera: camera_utils.Camera) -> None:
        """Initialize and load ASMNet model.

        Args:
            config: ASMNet configuration. See ASMNet.Config for more information.
            camera: Camera used for the input image.
        """
        config = yoco.load_config(config, current_dict=ASMNet.default_config)
        self._parse_config(config)
        self._camera = camera

    def _parse_config(self, config: Config) -> None:
        self._device = config["device"]
        self._weights_dir_path = utils.resolve_path(config["models_dir"])
        self._asm_params_dir_path = utils.resolve_path(config["asm_params_dir"])
        self._weights_url = config["weights_url"]
        self._check_paths()
        synset_names = ["placeholder"] + config["categories"]  # first will be ignored
        self._asmds = asmnet.cr6d_utils.load_asmds(
            self._asm_params_dir_path, synset_names
        )
        self._models = asmnet.cr6d_utils.load_models_release(
            self._weights_dir_path,
            synset_names,
            config["deformation_dimension"],
            config["num_points"],
            self._device,
        )
        self._num_points = config["num_points"]
        self._use_mean_shape = config["use_mean_shape"]
        self._use_icp = config["use_icp"]

    def _check_paths(self) -> None:
        if not os.path.exists(self._weights_dir_path) or not os.path.exists(
            self._asm_params_dir_path
        ):
            print("ASM-Net model weights not found, do you want to download to ")
            print("  ", self._weights_dir_path)
            print("  ", self._asm_params_dir_path)
            while True:
                decision = input("(Y/n) ").lower()
                if decision == "" or decision == "y":
                    self._download_weights()
                    break
                elif decision == "n":
                    print("ASM-Net model weights not found. Aborting.")
                    exit(0)

    def _download_weights(self) -> None:
        download_dir_path = tempfile.mkdtemp()
        zip_file_path = os.path.join(download_dir_path, "asmnetweights.zip")
        utils.download(
            self._weights_url,
            zip_file_path,
        )
        zip_file = zipfile.ZipFile(zip_file_path)
        zip_file.extractall(download_dir_path)
        zip_file.close()
        os.remove(zip_file_path)

        if not os.path.exists(self._asm_params_dir_path):
            os.makedirs(self._asm_params_dir_path, exist_ok=True)
            source_dir_path = os.path.join(download_dir_path, "params", "asm_params")
            file_names = os.listdir(source_dir_path)
            for fn in file_names:
                shutil.move(
                    os.path.join(source_dir_path, fn), self._asm_params_dir_path
                )

        if not os.path.exists(self._weights_dir_path):
            os.makedirs(self._weights_dir_path, exist_ok=True)
            source_dir_path = os.path.join(download_dir_path, "params", "weights")
            file_names = os.listdir(source_dir_path)
            for fn in file_names:
                shutil.move(os.path.join(source_dir_path, fn), self._weights_dir_path)

        shutil.rmtree(os.path.join(download_dir_path, "params"))

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_str: str,
    ) -> PredictionDict:
        """See CPASMethod.inference.

        Based on asmnet.ASM_Net.test_net_nocs2019_release
        """
        # torch -> numpy
        color_image = np.uint8(
            (color_image * 255).numpy()
        )  # (H, W, 3), uint8, 0-255, RGB
        depth_image = np.uint16((depth_image * 1000).numpy())  # (H, W), uint16, mm
        instance_mask = instance_mask.numpy()

        # Noise reduction + pointcloud generation
        masked_depth = depth_image * instance_mask
        masked_depth = asmnet.common3Dfunc.image_statistical_outlier_removal(
            masked_depth, factor=2.0
        )
        pcd_obj = asmnet.cr6d_utils.get_pcd_from_rgbd(
            color_image.copy(),
            masked_depth.copy(),
            self._camera.get_o3d_pinhole_camera_parameters().intrinsic,
        )
        [pcd_obj, _] = pcd_obj.remove_statistical_outlier(100, 2.0)
        pcd_in = copy.deepcopy(pcd_obj)
        pcd_c, offset = asmnet.common3Dfunc.centering(pcd_in)
        pcd_n, scale = asmnet.common3Dfunc.size_normalization(pcd_c)

        # o3d -> torch
        np_pcd = np.array(pcd_n.points)
        np_input = asmnet.cr6d_utils.random_sample(np_pcd, self._num_points)
        np_input = np_input.astype(np.float32)
        input_points = torch.from_numpy(np_input)

        # prepare input shape
        input_points = input_points.unsqueeze(0).transpose(2, 1).to(self._device)

        # evaluate model
        with torch.no_grad():
            dparam_pred, q_pred = self._models[category_str](input_points)
            dparam_pred = dparam_pred.cpu().numpy().squeeze()
            pred_rot = asmnet.cr6d_utils.quaternion2rotationPT(q_pred)
            pred_rot = pred_rot.cpu().numpy().squeeze()
            pred_dp_param = dparam_pred[:-1]  # deformation params
            pred_scaling_param = dparam_pred[-1]  # scale

            # get shape prediction
            pcd_pred = None
            if self._use_mean_shape:
                pcd_pred = self._asmds[category_str].deformation([0])
            else:
                pcd_pred = self._asmds[category_str].deformation(pred_dp_param)
                pcd_pred = pcd_pred.remove_statistical_outlier(20, 1.0)[0]
                pcd_pred.scale(pred_scaling_param, (0.0, 0.0, 0.0))

            metric_pcd = copy.deepcopy(pcd_pred)
            metric_pcd.scale(scale, (0.0, 0.0, 0.0))  # undo scale normalization

            # ICP
            pcd_pred_posed = copy.deepcopy(metric_pcd)
            pcd_pred_posed.rotate(pred_rot)  # rotate metric reconstruction
            pcd_pred_posed.translate(offset)  # move to center of cropped pcd
            pred_rt = np.identity(4)
            pred_rt[:3, :3] = pred_rot
            if self._use_icp:
                pcd_pred_posed_ds = pcd_pred_posed.voxel_down_sample(0.005)
                if len(pcd_pred_posed_ds.points) > 3:
                    # remove hidden points
                    pcd_pred_posed_visible = asmnet.common3Dfunc.applyHPR(
                        pcd_pred_posed_ds
                    )
                    pcd_in = pcd_in.voxel_down_sample(0.005)
                    reg_result = o3d.pipelines.registration.registration_icp(
                        pcd_pred_posed_visible, pcd_in, max_correspondence_distance=0.02
                    )
                    pcd_pred_posed = copy.deepcopy(pcd_pred_posed_ds).transform(
                        reg_result.transformation
                    )
                    pred_rt = np.dot(reg_result.transformation, pred_rt)
                else:
                    print(
                        "ASM-Net Warning: Couldn't perform ICP, too few points after"
                        "voxel down sampling"
                    )

            # center position
            maxb = pcd_pred_posed.get_max_bound()  # bbox max
            minb = pcd_pred_posed.get_min_bound()  # bbox min
            center = (maxb - minb) / 2 + minb  # bbox center
            pred_rt[:3, 3] = center.copy()

            position = torch.Tensor(pred_rt[:3, 3])
            orientation_q = torch.Tensor(
                Rotation.from_matrix(pred_rt[:3, :3]).as_quat()
            )
            reconstructed_points = torch.from_numpy(np.asarray(metric_pcd.points))

            # NOCS Object -> ShapeNet Object convention
            obj_fix = torch.tensor(
                [0.0, -1 / np.sqrt(2.0), 0.0, 1 / np.sqrt(2.0)]
            )  # CASS object to ShapeNet object
            orientation_q = quaternion_utils.quaternion_multiply(orientation_q, obj_fix)
            reconstructed_points = quaternion_utils.quaternion_apply(
                quaternion_utils.quaternion_invert(obj_fix),
                reconstructed_points,
            )
            extents, _ = reconstructed_points.abs().max(dim=0)
            extents *= 2.0

        return {
            "position": position,
            "orientation": orientation_q,
            "extents": extents,
            "reconstructed_pointcloud": reconstructed_points,
            "reconstructed_mesh": None,
        }
