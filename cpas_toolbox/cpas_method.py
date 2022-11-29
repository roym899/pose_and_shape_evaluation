"""This module defines the interface for categorical pose and shape estimation methods.

This module defines two classes: PredictionDict and CPASMethod. PredictionDict defines
the prediction produced by a CPASMethod. CPASMethod defines the interface used to
evaluate categorical pose and shape estimation methods.
"""
import copy
import os
import sys
import zipfile
from abc import ABC
from typing import List, Optional

if sys.version_info[0] >= 3 and sys.version_info[1] >= 8:
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

import numpy as np
import open3d as o3d
import torch
import yoco

from cpas_toolbox import (
    camera_utils,
    pointset_utils,
    quaternion_utils,
    utils,
)


class PredictionDict(TypedDict):
    """Pose and shape prediction.

    Attributes:
        position:
            Position of object center in camera frame. OpenCV convention. Shape (3,).
        orientation:
            Orientation of object in camera frame. OpenCV convention.
            Scalar-last quaternion, shape (4,).
        extents:
            Bounding box side lengths., shape (3,).
        reconstructed_pointcloud:
            Reconstructed pointcloud in object frame.
            None if method does not perform reconstruction.
        reconstructed_mesh:
            Reconstructed mesh in object frame.
            None if method does not perform reconstruction.
    """

    position: torch.Tensor
    orientation: torch.Tensor
    extents: torch.Tensor
    reconstructed_pointcloud: Optional[torch.Tensor]
    reconstructed_mesh: Optional[o3d.geometry.TriangleMesh]


class CPASMethod(ABC):
    """Interface class for categorical pose and shape estimation methods."""

    def inference(
        self,
        color_image: torch.Tensor,
        depth_image: torch.Tensor,
        instance_mask: torch.Tensor,
        category_str: str,
    ) -> PredictionDict:
        """Run a method to predict pose and shape of an object.

        Args:
            color_image: The color image, shape (H, W, 3), RGB, 0-1, float.
            depth_image: The depth image, shape (H, W), meters, float.
            instance_mask: Mask of object of interest. (H, W), bool.
            category_str: The category of the object.
        """
        pass


# class CASSWrapper(MethodWrapper):
#     """Wrapper class for CASS."""

#     class Config(TypedDict):
#         """Configuration dictionary for CASS.

#         Attributes:
#             model: Path to model.
#             device: Device string for the model.
#         """

#         model: str

#     default_config: Config = {
#         "model": None,
#     }

#     def __init__(self, config: Config, camera: camera_utils.Camera) -> None:
#         """Initialize and load CASS model.

#         Args:
#             config: CASS configuration. See CASSWrapper.Config for more information.
#             camera: Camera used for the input image.
#         """
#         config = yoco.load_config(config, current_dict=CASSWrapper.default_config)
#         self._parse_config(config)
#         self._camera = camera

#     def _parse_config(self, config: Config) -> None:
#         self._device = config["device"]
#         self._model_path = utils.resolve_path(config["model"])
#         self._check_paths()
#         self._cass = cass.CASS(
#             num_points=config["num_points"], num_obj=config["num_objects"]
#         )
#         self._num_points = config["num_points"]
#         self._cass.load_state_dict(torch.load(self._model_path))
#         self._cass.to(config["device"])
#         self._cass.eval()

#     def _check_paths(self) -> None:
#         if not os.path.exists(self._model_path):
#             print("CASS model weights not found, do you want to download to ")
#             print("  ", self._model_path)
#             while True:
#                 decision = input("(Y/n) ").lower()
#                 if decision == "" or decision == "y":
#                     self._download_weights()
#                     break
#                 elif decision == "n":
#                     print("CASS model weights not found. Aborting.")
#                     exit(0)

#     def _download_weights(self) -> None:
#         if not os.path.exists(self._model_path):
#             os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
#             utils.download(
#                 "https://drive.google.com/u/0/uc?id=14K1a-Ft-YO9dUREEXxmWqF2ruUP4p7BZ&"
#                 "export=download",
#                 self._model_path,
#             )

#     def inference(
#         self,
#         color_image: torch.Tensor,
#         depth_image: torch.Tensor,
#         instance_mask: torch.Tensor,
#         category_str: str,
#     ) -> PredictionDict:
#         """See MethodWrapper.inference.

#         Based on cass.tools.eval.
#         """
#         # get bounding box
#         valid_mask = (depth_image != 0) * instance_mask
#         rmin, rmax, cmin, cmax = cass.get_bbox(valid_mask.numpy())
#         bb_mask = torch.zeros_like(depth_image)
#         bb_mask[rmin:rmax, cmin:cmax] = 1.0

#         # prepare image crop
#         color_input = torch.flip(color_image, (2,)).permute([2, 0, 1])  # RGB -> BGR
#         color_input = color_input[:, rmin:rmax, cmin:cmax]  # bb crop
#         color_input = color_input.unsqueeze(0)  # add batch dim
#         color_input = TF.normalize(
#             color_input, mean=[0.51, 0.47, 0.44], std=[0.29, 0.27, 0.28]
#         )

#         # prepare points (fixed number of points, randomly picked)
#         point_indices = valid_mask.nonzero()
#         if len(point_indices) > self._num_points:
#             subset = np.random.choice(
#                 len(point_indices), replace=False, size=self._num_points
#             )
#             point_indices = point_indices[subset]
#         depth_mask = torch.zeros_like(depth_image)
#         depth_mask[point_indices[:, 0], point_indices[:, 1]] = 1.0
#         cropped_depth_mask = depth_mask[rmin:rmax, cmin:cmax]
#         point_indices_input = cropped_depth_mask.flatten().nonzero()[:, 0]

#         # prepare pointcloud
#         points = pointset_utils.depth_to_pointcloud(
#             depth_image,
#             self._camera,
#             normalize=False,
#             mask=depth_mask,
#             convention="opencv",
#         )
#         if len(points) < self._num_points:
#             wrap_indices = np.pad(
#                 np.arange(len(points)), (0, self._num_points - len(points)), mode="wrap"
#             )
#             points = points[wrap_indices]
#             point_indices_input = point_indices_input[wrap_indices]

#         # x, y inverted for some reason...
#         points[:, 0] *= -1
#         points[:, 1] *= -1
#         points = points.unsqueeze(0)
#         point_indices_input = point_indices_input.unsqueeze(0)

#         # move inputs to device
#         color_input = color_input.to(self._device)
#         points = points.to(self._device)
#         point_indices_input = point_indices_input.to(self._device)

#         category_str_to_id = {
#             "bottle": 0,
#             "bowl": 1,
#             "camera": 2,
#             "can": 3,
#             "laptop": 4,
#             "mug": 5,
#         }
#         category_id = category_str_to_id[category_str]

#         # CASS model uses 0-indexed categories, same order as NOCSDataset
#         category_index = torch.tensor([category_id], device=self._device)

#         # Call CASS network
#         folding_encode = self._cass.foldingnet.encode(
#             color_input, points, point_indices_input
#         )
#         posenet_encode = self._cass.estimator.encode(
#             color_input, points, point_indices_input
#         )
#         pred_r, pred_t, pred_c = self._cass.estimator.pose(
#             torch.cat([posenet_encode, folding_encode], dim=1), category_index
#         )
#         reconstructed_points = self._cass.foldingnet.recon(folding_encode)[0]

#         # Postprocess outputs
#         reconstructed_points = reconstructed_points.view(-1, 3).cpu()
#         pred_c = pred_c.view(1, self._num_points)
#         _, max_index = torch.max(pred_c, 1)
#         pred_t = pred_t.view(self._num_points, 1, 3)
#         orientation_q = pred_r[0][max_index[0]].view(-1).cpu()
#         points = points.view(self._num_points, 1, 3)
#         position = (points + pred_t)[max_index[0]].view(-1).cpu()
#         # output is scalar-first -> scalar-last
#         orientation_q = torch.tensor([*orientation_q[1:], orientation_q[0]])

#         # Flip x and y axis of position and orientation (undo flipping of points)
#         # (x-left, y-up, z-forward) convention -> OpenCV convention
#         position[0] *= -1
#         position[1] *= -1
#         cam_fix = torch.tensor([0.0, 0.0, 1.0, 0.0])
#         # NOCS Object -> ShapeNet Object convention
#         obj_fix = torch.tensor(
#             [0.0, -1 / np.sqrt(2.0), 0.0, 1 / np.sqrt(2.0)]
#         )  # CASS object to ShapeNet object
#         orientation_q = quaternion_utils.quaternion_multiply(cam_fix, orientation_q)
#         orientation_q = quaternion_utils.quaternion_multiply(orientation_q, obj_fix)
#         reconstructed_points = quaternion_utils.quaternion_apply(
#             quaternion_utils.quaternion_invert(obj_fix),
#             reconstructed_points,
#         )

#         # TODO refinement code from cass.tools.eval? (not mentioned in paper??)

#         extents, _ = reconstructed_points.abs().max(dim=0)
#         extents *= 2.0

#         # pointset_utils.visualize_pointset(reconstructed_points)
#         return {
#             "position": position.detach(),
#             "orientation": orientation_q.detach(),
#             "extents": extents.detach(),
#             "reconstructed_pointcloud": reconstructed_points.detach(),
#             "reconstructed_mesh": None,
#         }


# class ASMNetWrapper:
#     """Wrapper class for ASMNet."""

#     class Config(TypedDict):
#         """Configuration dictionary for ASMNet.

#         Attributes:
#             model: Path to model.
#             device: Device string for the model.
#             models_folder:
#                 Path to folder containing model parameters.
#                 Must contain the following folder structure:
#                     {models_folder}/{category_0}/model.pth
#                     ...
#             asm_params_folder:
#                 Path to folder containing ASM parameters.
#                 Must contain the following folder structure:
#                     {asm_params_folder}/{category_0}/train/info.npz
#                     ...
#             categories:
#                 List of categories. Each category requires corresponding folder with
#                 model.pth and info.npz. See models_folder and asm_params_folder.
#             num_points: Number of input poins.
#             deformation_dimension: Number of deformation parameters.
#             use_mean_shape:
#                 Whether the mean shape (0) or the predicted shape deformation should
#                 be used.
#             use_icp: Whether to use ICP to refine the pose.
#         """

#         models_folder: str
#         asm_params_folder: str
#         device: str
#         categories: List[str]
#         num_points: int
#         deformation_dimension: int
#         use_mean_shape: bool
#         use_icp: bool

#     default_config: Config = {
#         "model_params_folder": None,
#         "asm_params_folder": None,
#         "device": "cuda",
#         "categories": [],
#         "num_points": 800,
#         "deformation_dimension": 3,
#         "use_mean_shape": False,
#         "use_icp": True,
#     }

#     def __init__(self, config: Config, camera: camera_utils.Camera) -> None:
#         """Initialize and load ASMNet model.

#         Args:
#             config: ASMNet configuration. See ASMNetWrapper.Config for more information.
#             camera: Camera used for the input image.
#         """
#         config = yoco.load_config(config, current_dict=ASMNetWrapper.default_config)
#         self._parse_config(config)
#         self._camera = camera

#     def _parse_config(self, config: Config) -> None:
#         self._device = config["device"]
#         self._weights_folder = utils.resolve_path(config["models_folder"])
#         self._asm_params_folder = utils.resolve_path(config["asm_params_folder"])
#         self._check_paths()
#         synset_names = ["placeholder"] + config["categories"]  # first will be ignored
#         self._asmds = asmnet.cr6d_utils.load_asmds(
#             self._asm_params_folder, synset_names
#         )
#         self._models = asmnet.cr6d_utils.load_models_release(
#             self._weights_folder,
#             synset_names,
#             config["deformation_dimension"],
#             config["num_points"],
#             self._device,
#         )
#         self._num_points = config["num_points"]
#         self._use_mean_shape = config["use_mean_shape"]
#         self._use_icp = config["use_icp"]

#     def _check_paths(self) -> None:
#         if not os.path.exists(self._weights_folder) or not os.path.exists(
#             self._asm_params_folder
#         ):
#             print("ASM-Net model weights not found, do you want to download to ")
#             print("  ", self._weights_folder)
#             print("  ", self._asm_params_folder)
#             while True:
#                 decision = input("(Y/n) ").lower()
#                 if decision == "" or decision == "y":
#                     self._download_weights()
#                     break
#                 elif decision == "n":
#                     print("ASM-Net model weights not found. Aborting.")
#                     exit(0)

#     def _download_weights(self) -> None:
#         download_folder = "./"
#         zip_path = os.path.join(download_folder, "asmnetweights.zip")
#         utils.download(
#             "https://drive.google.com/u/0/uc?id=1fxc9UoRhfTsoV3ZML3Mx_mc79904zpcx"
#             "&export=download",
#             zip_path,
#         )
#         z = zipfile.ZipFile(zip_path)
#         z.extractall(download_folder)
#         z.close()
#         os.remove(zip_path)

#         if not os.path.exists(self._asm_params_folder):
#             os.makedirs(self._asm_params_folder, exist_ok=True)
#             source_dir = os.path.join(download_folder, "params", "asm_params")
#             file_names = os.listdir(source_dir)
#             for fn in file_names:
#                 shutil.move(os.path.join(source_dir, fn), self._asm_params_folder)

#         if not os.path.exists(self._weights_folder):
#             os.makedirs(self._weights_folder, exist_ok=True)
#             source_dir = os.path.join(download_folder, "params", "weights")
#             file_names = os.listdir(source_dir)
#             for fn in file_names:
#                 shutil.move(os.path.join(source_dir, fn), self._weights_folder)

#         shutil.rmtree(os.path.join(download_folder, "params"))

#     def inference(
#         self,
#         color_image: torch.Tensor,
#         depth_image: torch.Tensor,
#         instance_mask: torch.Tensor,
#         category_str: str,
#     ) -> PredictionDict:
#         """See MethodWrapper.inference.

#         Based on asmnet.ASM_Net.test_net_nocs2019_release
#         """
#         # torch -> numpy
#         color_image = np.uint8(
#             (color_image * 255).numpy()
#         )  # (H, W, 3), uint8, 0-255, RGB
#         depth_image = np.uint16((depth_image * 1000).numpy())  # (H, W), uint16, mm
#         instance_mask = instance_mask.numpy()

#         # Noise reduction + pointcloud generation
#         masked_depth = depth_image * instance_mask
#         masked_depth = asmnet.common3Dfunc.image_statistical_outlier_removal(
#             masked_depth, factor=2.0
#         )
#         pcd_obj = asmnet.cr6d_utils.get_pcd_from_rgbd(
#             color_image.copy(),
#             masked_depth.copy(),
#             self._camera.get_o3d_pinhole_camera_parameters().intrinsic,
#         )
#         [pcd_obj, _] = pcd_obj.remove_statistical_outlier(100, 2.0)
#         pcd_in = copy.deepcopy(pcd_obj)
#         pcd_c, offset = asmnet.common3Dfunc.centering(pcd_in)
#         pcd_n, scale = asmnet.common3Dfunc.size_normalization(pcd_c)

#         # o3d -> torch
#         np_pcd = np.array(pcd_n.points)
#         np_input = asmnet.cr6d_utils.random_sample(np_pcd, self._num_points)
#         np_input = np_input.astype(np.float32)
#         input_points = torch.from_numpy(np_input)

#         # prepare input shape
#         input_points = input_points.unsqueeze(0).transpose(2, 1).to(self._device)

#         # evaluate model
#         with torch.no_grad():
#             dparam_pred, q_pred = self._models[category_str](input_points)
#             dparam_pred = dparam_pred.cpu().numpy().squeeze()
#             pred_rot = asmnet.cr6d_utils.quaternion2rotationPT(q_pred)
#             pred_rot = pred_rot.cpu().numpy().squeeze()
#             pred_dp_param = dparam_pred[:-1]  # deformation params
#             pred_scaling_param = dparam_pred[-1]  # scale

#             # get shape prediction
#             pcd_pred = None
#             if self._use_mean_shape:
#                 pcd_pred = self._asmds[category_str].deformation([0])
#             else:
#                 pcd_pred = self._asmds[category_str].deformation(pred_dp_param)
#                 pcd_pred = pcd_pred.remove_statistical_outlier(20, 1.0)[0]
#                 pcd_pred.scale(pred_scaling_param, (0.0, 0.0, 0.0))

#             metric_pcd = copy.deepcopy(pcd_pred)
#             metric_pcd.scale(scale, (0.0, 0.0, 0.0))  # undo scale normalization

#             # ICP
#             pcd_pred_posed = copy.deepcopy(metric_pcd)
#             pcd_pred_posed.rotate(pred_rot)  # rotate metric reconstruction
#             pcd_pred_posed.translate(offset)  # move to center of cropped pcd
#             pred_rt = np.identity(4)
#             pred_rt[:3, :3] = pred_rot
#             if self._use_icp:
#                 pcd_pred_posed_ds = pcd_pred_posed.voxel_down_sample(0.005)
#                 if len(pcd_pred_posed_ds.points) > 3:
#                     # remove hidden points
#                     pcd_pred_posed_visible = asmnet.common3Dfunc.applyHPR(
#                         pcd_pred_posed_ds
#                     )
#                     pcd_in = pcd_in.voxel_down_sample(0.005)
#                     reg_result = o3d.pipelines.registration.registration_icp(
#                         pcd_pred_posed_visible, pcd_in, max_correspondence_distance=0.02
#                     )
#                     pcd_pred_posed = copy.deepcopy(pcd_pred_posed_ds).transform(
#                         reg_result.transformation
#                     )
#                     pred_rt = np.dot(reg_result.transformation, pred_rt)
#                 else:
#                     print(
#                         "ASM-Net Warning: Couldn't perform ICP, too few points after"
#                         "voxel down sampling"
#                     )

#             # center position
#             maxb = pcd_pred_posed.get_max_bound()  # bbox max
#             minb = pcd_pred_posed.get_min_bound()  # bbox min
#             center = (maxb - minb) / 2 + minb  # bbox center
#             pred_rt[:3, 3] = center.copy()

#             position = torch.Tensor(pred_rt[:3, 3])
#             orientation_q = torch.Tensor(
#                 Rotation.from_matrix(pred_rt[:3, :3]).as_quat()
#             )
#             reconstructed_points = torch.from_numpy(np.asarray(metric_pcd.points))

#             # NOCS Object -> ShapeNet Object convention
#             obj_fix = torch.tensor(
#                 [0.0, -1 / np.sqrt(2.0), 0.0, 1 / np.sqrt(2.0)]
#             )  # CASS object to ShapeNet object
#             orientation_q = quaternion_utils.quaternion_multiply(orientation_q, obj_fix)
#             reconstructed_points = quaternion_utils.quaternion_apply(
#                 quaternion_utils.quaternion_invert(obj_fix),
#                 reconstructed_points,
#             )
#             extents, _ = reconstructed_points.abs().max(dim=0)
#             extents *= 2.0

#         return {
#             "position": position,
#             "orientation": orientation_q,
#             "extents": extents,
#             "reconstructed_pointcloud": reconstructed_points,
#             "reconstructed_mesh": None,
#         }
