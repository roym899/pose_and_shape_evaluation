"""This module defines iCaps interface.

Method is described in iCaps Iterative Category-Level Object Pose and Shape Estimation,
Deng, 2022.

Implementation based on https://github.com/aerogjy/iCaps
"""
import copy
import os
import shutil
import tarfile
import tempfile
from typing import TypedDict

import numpy as np
import scipy
import torch
import yoco
from scipy.spatial.transform import Rotation

from cpas_toolbox import camera_utils, quaternion_utils, utils
from cpas_toolbox.cpas_method import CPASMethod, PredictionDict

from . import _icaps as icaps


class ICaps(CPASMethod):
    """Wrapper class for iCaps."""

    class Config(TypedDict):
        """Configuration dictionary for iCaps.

        Attributes:
            pf_config_folder:
                Particle filter configuration folder for iCaps. Must contain one yml
                file for each supported category_str.
            deepsdf_checkpoint_folder: Folder containing DeepSDF checkpoints.
            latentnet_checkpoint_folder: Folder containing LatentNet checkpoints.
            aae_checkpoint_folder: Folder containing auto-encoder checkpoints.
            checkpoints_url:
                URL to download checkpoints from if checkpoint folders are empty or do
                not exist yet (assumed to be tar file).
        """

        pf_config_folder: str
        deepsdf_checkpoint_folder: str
        latentnet_checkpoint_folder: str
        aae_checkpoint_folder: str
        checkpoints_url: str

    default_config: Config = {
        "pf_config_folder": None,
        "deepsdf_checkpoint_folder": None,
        "latentnet_checkpoint_folder": None,
        "aae_checkpoint_folder": None,
        "checkpoints_url": None,
    }

    def __init__(self, config: Config, camera: camera_utils.Camera) -> None:
        """Initialize and load ICaps models.

        Args:
            config: iCaps configuration. See ICaps.Config for more information.
            camera: Camera used for the input image.
        """
        config = yoco.load_config(config, current_dict=ICaps.default_config)
        self._parse_config(config)
        self._camera = camera

    def _parse_config(self, config: Config) -> None:
        self._num_points = 10000
        # TODO fix this
        self._category_str_to_ckpt_folder = {
            "bottle": "bottle20200608T172228_default",
            "bowl": "bowl20200603T175721_default",
            "camera": "camera20200603T175729_default",
            "can": "can20200605T201536_default",
            "laptop": "laptop20200605T205824_default",
            "mug": "mug20200529T111737_default",
        }

        self._checkpoints_url = config["checkpoints_url"]
        self._pose_rbpfs = {}
        pf_cfg_folder = utils.resolve_path(
            config["pf_config_folder"],
            search_paths=[
                ".",
                "~/.cpas_toolbox",
                os.path.join(os.path.dirname(__file__), "config"),
                os.path.dirname(__file__),
            ],
        )
        self._deepsdf_ckp_folder = utils.resolve_path(
            config["deepsdf_checkpoint_folder"]
        )
        self._latentnet_ckp_folder = utils.resolve_path(
            config["latentnet_checkpoint_folder"]
        )
        self._aae_ckp_folder = utils.resolve_path(config["aae_checkpoint_folder"])
        self._check_paths()

        for (
            category_str,
            aae_category_folder,
        ) in self._category_str_to_ckpt_folder.items():
            full_ckpt_folder = os.path.join(self._aae_ckp_folder, aae_category_folder)
            train_cfg_file = os.path.join(full_ckpt_folder, "config.yml")
            icaps.icaps_config.cfg_from_file(train_cfg_file)
            test_cfg_file = os.path.join(pf_cfg_folder, category_str + ".yml")
            icaps.icaps_config.cfg_from_file(test_cfg_file)
            obj_list = icaps.icaps_config.cfg.TEST.OBJECTS
            cfg_list = []
            cfg_list.append(copy.deepcopy(icaps.icaps_config.cfg))

            self._pose_rbpfs[category_str] = icaps.PoseRBPF(
                obj_list,
                cfg_list,
                full_ckpt_folder,
                self._deepsdf_ckp_folder,
                self._latentnet_ckp_folder,
            )
            self._pose_rbpfs[category_str].set_target_obj(
                icaps.icaps_config.cfg.TEST.OBJECTS[0]
            )

    def _check_paths(self) -> None:
        path_exists = (
            os.path.exists(p)
            for p in [
                self._aae_ckp_folder,
                self._deepsdf_ckp_folder,
                self._latentnet_ckp_folder,
            ]
        )
        if not all(path_exists):
            print("iCaps model weights not found, do you want to download to ")
            print("  ", self._aae_ckp_folder)
            print("  ", self._deepsdf_ckp_folder)
            print("  ", self._latentnet_ckp_folder)
            while True:
                decision = input("(Y/n) ").lower()
                if decision == "" or decision == "y":
                    self._download_weights()
                    break
                elif decision == "n":
                    print("iCaps model weights not found. Aborting.")
                    exit(0)

    def _download_weights(self) -> None:
        download_dir_path = tempfile.mkdtemp()
        tar_file_path = os.path.join(download_dir_path, "temp")
        print(self._checkpoints_url, tar_file_path)
        utils.download(
            self._checkpoints_url,
            tar_file_path,
        )
        tar_file = tarfile.open(tar_file_path)
        print("Extracting weights... (this might take a while)")
        tar_file.extractall(download_dir_path)
        if not os.path.exists(self._latentnet_ckp_folder):
            src_dir = os.path.join(download_dir_path, "checkpoints", "latentnet_ckpts")
            shutil.move(src_dir, self._latentnet_ckp_folder)
        if not os.path.exists(self._deepsdf_ckp_folder):
            src_dir = os.path.join(download_dir_path, "checkpoints", "deepsdf_ckpts")
            shutil.move(src_dir, self._deepsdf_ckp_folder)
        if not os.path.exists(self._aae_ckp_folder):
            src_dir = os.path.join(download_dir_path, "checkpoints", "aae_ckpts")
            shutil.move(src_dir, self._aae_ckp_folder)

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_str: str,
    ) -> PredictionDict:
        """See MethodWrapper.inference.

        Based on icaps.pose_rbpf.pose_rbps.PoseRBPF.run_nocs_dataset
        """
        # prepare data as expected by iCaps functions (same as nocs_real_dataset)
        color_image = color_image * 255  # see icaps.datasets.nocs_real_dataset l71
        depth_image = depth_image.unsqueeze(2)  # (...)nocs_real_dataset l79
        instance_mask = instance_mask.float()  # (...)nocs_real_dataset l100
        intrinsics = torch.eye(3)
        fx, fy, cx, cy, _ = self._camera.get_pinhole_camera_parameters(pixel_center=0.0)
        intrinsics[0, 0] = fx
        intrinsics[1, 1] = fy
        intrinsics[0, 2] = cx
        intrinsics[1, 2] = cy
        x1 = min(instance_mask.nonzero()[:, 1]).item()
        y1 = min(instance_mask.nonzero()[:, 0]).item()
        x2 = max(instance_mask.nonzero()[:, 1]).item()
        y2 = max(instance_mask.nonzero()[:, 0]).item()
        bbox = [y1, y2, x1, x2]

        # from here follow icaps.pose_rbpf.pose_rbps.PoseRBPF.run_nocs_dataset
        pose_rbpf = self._pose_rbpfs[category_str]
        pose_rbpf.reset()  # like init but without loading models
        self._pose_rbpfs[category_str].set_target_obj(category_str)

        pose_rbpf.data_intrinsics = intrinsics.numpy()
        pose_rbpf.intrinsics = intrinsics.numpy()
        pose_rbpf.target_obj_cfg.PF.FU = pose_rbpf.intrinsics[0, 0]
        pose_rbpf.target_obj_cfg.PF.FV = pose_rbpf.intrinsics[1, 1]
        pose_rbpf.target_obj_cfg.PF.U0 = pose_rbpf.intrinsics[0, 2]
        pose_rbpf.target_obj_cfg.PF.V0 = pose_rbpf.intrinsics[1, 2]

        pose_rbpf.data_with_est_center = False
        pose_rbpf.data_with_gt = False  # should this be False now?

        pose_rbpf.mask_raw = instance_mask[:, :].cpu().numpy()
        pose_rbpf.mask = scipy.ndimage.binary_erosion(
            pose_rbpf.mask_raw, iterations=2
        ).astype(pose_rbpf.mask_raw.dtype)

        pose_rbpf.prior_uv[0] = (bbox[2] + bbox[3]) / 2
        pose_rbpf.prior_uv[1] = (bbox[0] + bbox[1]) / 2

        # what is this ??
        if pose_rbpf.aae_full.angle_diff.shape[0] != 0:
            pose_rbpf.aae_full.angle_diff = np.array([])

        if pose_rbpf.target_obj_cfg.PF.USE_DEPTH:
            depth_data = depth_image
        else:
            depth_data = None

        try:
            pose_rbpf.initialize_poserbpf(
                color_image,
                pose_rbpf.data_intrinsics,
                pose_rbpf.prior_uv[:2],
                pose_rbpf.target_obj_cfg.PF.N_INIT,
                scale_prior=pose_rbpf.target_obj_cfg.PF.SCALE_PRIOR,
                depth=depth_data,
            )

            pose_rbpf.process_poserbpf(
                color_image,
                intrinsics.unsqueeze(0),
                depth=depth_data,
                run_deep_sdf=False,
            )
            # 3 * 50 iters by default
            pose_rbpf.refine_pose_and_shape(depth_data, intrinsics.unsqueeze(0))

            position_cv = torch.tensor(pose_rbpf.rbpf.trans_bar)
            orientation_q = torch.Tensor(
                Rotation.from_matrix(pose_rbpf.rbpf.rot_bar).as_quat()
            )

            # NOCS Object -> ShapeNet Object convention
            obj_fix = torch.tensor(
                [0.0, -1 / np.sqrt(2.0), 0.0, 1 / np.sqrt(2.0)]
            )  # CASS object to ShapeNet object
            orientation_q = quaternion_utils.quaternion_multiply(orientation_q, obj_fix)

            orientation_cv = orientation_q
            extents = torch.tensor([0.5, 0.5, 0.5])

            point_set = pose_rbpf.evaluator.latent_vec_to_points(
                pose_rbpf.latent_vec_refine,
                N=64,
                num_points=self._num_points,
                silent=True,
            )
            if point_set is None:
                point_set = torch.tensor([[0.0, 0.0, 0.0]])  # failed / no isosurface
            else:
                point_set *= pose_rbpf.size_est / pose_rbpf.ratio

            reconstructed_points = torch.tensor(point_set)

            extents, _ = reconstructed_points.abs().max(dim=0)
            extents *= 2.0
            return {
                "position": position_cv.detach().cpu(),
                "orientation": orientation_cv.detach().cpu(),
                "extents": extents.detach().cpu(),
                "reconstructed_pointcloud": reconstructed_points,
                "reconstructed_mesh": None,  # TODO if time
            }
        except KeyboardInterrupt:
            print("===PROBLEM DETECTED WITH ICAPS, RETURNING NO PREDICTION INSTEAD===")
            return {
                "position": torch.tensor([0.0, 0.0, 0.0]),
                "orientation": torch.tensor([0.0, 0.0, 0.0, 1.0]),
                "extents": torch.tensor([0.5, 0.5, 0.5]),
                "reconstructed_pointcloud": torch.tensor([[0.0, 0.0, 0.0]]),
                "reconstructed_mesh": None,  # TODO if time
            }
