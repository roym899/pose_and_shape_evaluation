"""Module providing dataset class for annotated Redwood dataset."""
import json
import os
from typing import TypedDict, Optional
import zipfile

from scipy.spatial.transform import Rotation
import numpy as np
import open3d as o3d
import torch
from PIL import Image
import yoco

from cpas_toolbox import camera_utils, pointset_utils, quaternion_utils, utils


class AnnotatedRedwoodDataset(torch.utils.data.Dataset):
    """Dataset class for annotated Redwood dataset.

    Data can be found here:
    http://redwood-data.org/3dscan/index.html

    Annotations are of repo.

    Expected directory format:
        {root_dir}/{category_str}/rgbd/{sequence_id}/...
        {ann_dir}/{sequence_id}.obj
        {ann_dir}/annotations.json
    """

    num_categories = 3
    category_id_to_str = {
        0: "bottle",
        1: "bowl",
        2: "mug",
    }
    category_str_to_id = {v: k for k, v in category_id_to_str.items()}

    class Config(TypedDict, total=False):
        """Configuration dictionary for annoated Redwood dataset.

        Attributes:
            root_dir: See AnnotatedRedwoodDataset docstring.
            ann_dir: See AnnotatedRedwoodDataset docstring.
            mask_pointcloud: Whether the returned pointcloud will be masked.
            normalize_pointcloud:
                Whether the returned pointcloud and position will be normalized, such
                that pointcloud centroid is at the origin.
            scale_convention:
                Which scale is returned. The following strings are supported:
                    "diagonal":
                        Length of bounding box' diagonal. This is what NOCS uses.
                    "max": Maximum side length of bounding box.
                    "half_max": Half maximum side length of bounding box.
                    "full": Bounding box side lengths. Shape (3,).
            camera_convention:
                Which camera convention is used for position and orientation. One of:
                    "opengl": x right, y up, z back
                    "opencv": x right, y down, z forward
                Note that this does not influence how the dataset is processed, only the
                returned position and quaternion.
            orientation_repr:
                Which orientation representation is used. Currently only "quaternion"
                supported.
            remap_y_axis:
                If not None, the Redwood y-axis will be mapped to the provided axis.
                Resulting coordinate system will always be right-handed.
                This is typically the up-axis.
                Note that NOCS object models are NOT aligned the same as ShapeNetV2.
                To get ShapeNetV2 alignment: -y
                One of: "x", "y", "z", "-x", "-y", "-z"
            remap_x_axis:
                If not None, the original x-axis will be mapped to the provided axis.
                Resulting coordinate system will always be right-handed.
                Note that NOCS object models are NOT aligned the same as ShapeNetV2.
                To get ShapeNetV2 alignment: z
                One of: "x", "y", "z", "-x", "-y", "-z"
            category_str:
                If not None, only samples from the matching category will be returned.
                See AnnotatedRedwoodDataset.category_id_to_str for admissible category
                strings.
        """

        root_dir: str
        ann_dir: str
        split: str
        mask_pointcloud: bool
        normalize_pointcloud: bool
        scale_convention: str
        camera_convention: str
        orientation_repr: str
        orientation_grid_resolution: int
        remap_y_axis: Optional[str]
        remap_x_axis: Optional[str]
        category_str: Optional[str]

    default_config: Config = {
        "root_dir": None,
        "ann_dir": None,
        "mask_pointcloud": False,
        "normalize_pointcloud": False,
        "camera_convention": "opengl",
        "scale_convention": "half_max",
        "orientation_repr": "quaternion",
        "orientation_grid_resolution": None,
        "category_str": None,
        "remap_y_axis": None,
        "remap_x_axis": None,
    }

    def __init__(
        self,
        config: Config,
    ) -> None:
        """Initialize the dataset.

        Args:
            config:
                Configuration dictionary of dataset. Provided dictionary will be merged
                with default_dict. See AnnotatedRedwoodDataset.Config for keys.
        """
        config = yoco.load_config(
            config, current_dict=AnnotatedRedwoodDataset.default_config
        )
        self._root_dir = utils.resolve_path(config["root_dir"])
        self._ann_dir = utils.resolve_path(config["ann_dir"])
        self._check_dirs()
        self._camera_convention = config["camera_convention"]
        self._mask_pointcloud = config["mask_pointcloud"]
        self._normalize_pointcloud = config["normalize_pointcloud"]
        self._scale_convention = config["scale_convention"]
        self._remap_y_axis = config["remap_y_axis"]
        self._remap_x_axis = config["remap_x_axis"]
        self._orientation_repr = config["orientation_repr"]
        self._load_annotations()
        self._camera = camera_utils.Camera(
            width=640, height=480, fx=525, fy=525, cx=319.5, cy=239.5
        )

    def _check_dirs(self) -> None:
        if os.path.exists(self._root_dir) and os.path.exists(self._ann_dir):
            pass
        else:
            print(
                "REDWOOD75 dataset not found, do you want to download it into the "
                "following directories:"
            )
            print("  ", self._root_dir)
            print("  ", self._ann_dir)
            while True:
                decision = input("(Y/n) ").lower()
                if decision == "" or decision == "y":
                    self._download_dataset()
                    break
                elif decision == "n":
                    print("Dataset not found. Aborting.")
                    exit(0)

    def _download_dataset(self) -> None:
        # Download anns
        if not os.path.exists(self._ann_dir):
            zip_path = os.path.join(self._ann_dir, "redwood75.zip")
            os.makedirs(self._ann_dir, exist_ok=True)
            url = (
                "https://drive.google.com/u/0/uc?id=1PMvIblsXWDxEJykVwhUk_QEjy4_bmDU"
                "-&export=download"
            )
            utils.download(url, zip_path)
            z = zipfile.ZipFile(zip_path)
            z.extractall(os.path.join(self._ann_dir, ".."))
            z.close()
            os.remove(zip_path)

        ann_json = os.path.join(self._ann_dir, "annotations.json")
        with open(ann_json, "r") as f:
            anns_dict = json.load(f)

        baseurl = "https://s3.us-west-1.wasabisys.com/redwood-3dscan/rgbd/"
        for seq_id in anns_dict.keys():
            download_dir = os.path.join(self._root_dir, anns_dict[seq_id]["category"])
            os.makedirs(download_dir, exist_ok=True)
            zip_path = os.path.join(download_dir, f"{seq_id}.zip")
            os.makedirs(os.path.dirname(zip_path), exist_ok=True)
            utils.download(baseurl + f"{seq_id}.zip", zip_path)
            z = zipfile.ZipFile(zip_path)
            out_folder = os.path.join(download_dir, "rgbd", seq_id)
            os.makedirs(out_folder, exist_ok=True)
            z.extractall(out_folder)
            z.close()
            os.remove(zip_path)

    def _load_annotations(self) -> None:
        """Load annotations into memory."""
        ann_json = os.path.join(self._ann_dir, "annotations.json")
        with open(ann_json, "r") as f:
            anns_dict = json.load(f)
        self._raw_samples = []
        for seq_id, seq_anns in anns_dict.items():
            for pose_ann in seq_anns["pose_anns"]:
                self._raw_samples.append(
                    self._create_raw_sample(seq_id, seq_anns, pose_ann)
                )

    def _create_raw_sample(
        self, seq_id: str, sequence_dict: dict, annotation_dict: dict
    ) -> dict:
        """Create raw sample from information in annotations file."""
        position = torch.tensor(annotation_dict["position"])
        orientation_q = torch.tensor(annotation_dict["orientation"])
        rgb_filename = annotation_dict["rgb_file"]
        depth_filename = annotation_dict["depth_file"]
        mesh_filename = sequence_dict["mesh"]
        mesh_path = os.path.join(self._ann_dir, mesh_filename)
        category_str = sequence_dict["category"]
        color_path = os.path.join(
            self._root_dir, category_str, "rgbd", seq_id, "rgb", rgb_filename
        )
        depth_path = os.path.join(
            self._root_dir, category_str, "rgbd", seq_id, "depth", depth_filename
        )
        extents = torch.tensor(sequence_dict["scale"]) * 2
        return {
            "position": position,
            "orientation_q": orientation_q,
            "extents": extents,
            "color_path": color_path,
            "depth_path": depth_path,
            "mesh_path": mesh_path,
            "category_str": category_str,
        }

    def __len__(self) -> int:
        """Return number of sample in dataset."""
        return len(self._raw_samples)

    def __getitem__(self, idx: int) -> dict:
        """Return a sample of the dataset.

        Args:
            idx: Index of the instance.

        Returns:
            Sample containing the following keys:
                "color"
                "depth"
                "mask"
                "pointset"
                "position"
                "orientation"
                "quaternion"
                "scale"
                "color_path"
                "obj_path"
                "category_id"
                "category_str"
        """
        raw_sample = self._raw_samples[idx]
        color = torch.from_numpy(
            np.asarray(Image.open(raw_sample["color_path"]), dtype=np.float32) / 255
        )
        depth = self._load_depth(raw_sample["depth_path"])
        instance_mask = self._compute_mask(depth, raw_sample)

        pointcloud_mask = instance_mask if self._mask_pointcloud else None
        pointcloud = pointset_utils.depth_to_pointcloud(
            depth,
            self._camera,
            mask=pointcloud_mask,
            convention=self._camera_convention,
        )

        # adjust camera convention for position, orientation and scale
        position = pointset_utils.change_position_camera_convention(
            raw_sample["position"], "opencv", self._camera_convention
        )

        # orientation / scale
        orientation_q, extents = self._change_axis_convention(
            raw_sample["orientation_q"], raw_sample["extents"]
        )
        orientation_q = pointset_utils.change_orientation_camera_convention(
            orientation_q, "opencv", self._camera_convention
        )
        orientation = self._quat_to_orientation_repr(orientation_q)
        scale = self._get_scale(extents)

        # normalize pointcloud & position
        if self._normalize_pointcloud:
            pointcloud, centroid = pointset_utils.normalize_points(pointcloud)
            position = position - centroid

        category_str = raw_sample["category_str"]
        sample = {
            "color": color,
            "depth": depth,
            "pointset": pointcloud,
            "mask": instance_mask,
            "position": position,
            "orientation": orientation,
            "quaternion": orientation_q,
            "scale": scale,
            "color_path": raw_sample["color_path"],
            "obj_path": raw_sample["mesh_path"],
            "category_id": self.category_str_to_id[category_str],
            "category_str": category_str,
        }
        return sample

    def _compute_mask(self, depth: torch.Tensor, raw_sample: dict) -> torch.Tensor:
        posed_mesh = o3d.io.read_triangle_mesh(raw_sample["mesh_path"])
        R = Rotation.from_quat(raw_sample["orientation_q"]).as_matrix()
        posed_mesh.rotate(R, center=np.array([0, 0, 0]))
        posed_mesh.translate(raw_sample["position"])
        posed_mesh.compute_vertex_normals()
        gt_depth = torch.from_numpy(_draw_depth_geometry(posed_mesh, self._camera))
        mask = gt_depth != 0
        # exclude occluded parts from mask
        mask[(depth != 0) * (depth < gt_depth - 0.01)] = 0
        return mask

    def _load_depth(self, depth_path: str) -> torch.Tensor:
        """Load depth from depth filepath."""
        depth = torch.from_numpy(
            np.asarray(Image.open(depth_path), dtype=np.float32) * 0.001
        )
        return depth

    def _get_scale(self, extents: torch.Tensor) -> float:
        """Return scale from stored sample data and extents."""
        if self._scale_convention == "diagonal":
            return torch.linalg.norm(extents)
        elif self._scale_convention == "max":
            return extents.max()
        elif self._scale_convention == "half_max":
            return 0.5 * extents.max()
        elif self._scale_convention == "full":
            return extents
        else:
            raise ValueError(
                f"Specified scale convention {self._scale_convention} not supported."
            )

    def _change_axis_convention(
        self, orientation_q: torch.Tensor, extents: torch.Tensor
    ) -> tuple:
        """Adjust up-axis for orientation and extents.

        Returns:
            Tuple of position, orienation_q and extents, with specified up-axis.
        """
        if self._remap_y_axis is None and self._remap_x_axis is None:
            return orientation_q, extents
        elif self._remap_y_axis is None or self._remap_x_axis is None:
            raise ValueError("Either both or none of remap_{y,x}_axis have to be None.")

        rotation_o2n = self._get_o2n_object_rotation_matrix()
        remapped_extents = torch.abs(torch.Tensor(rotation_o2n) @ extents)

        # quaternion so far: original -> camera
        # we want a quaternion: new -> camera
        rotation_n2o = rotation_o2n.T

        quaternion_n2o = torch.from_numpy(Rotation.from_matrix(rotation_n2o).as_quat())

        remapped_orientation_q = quaternion_utils.quaternion_multiply(
            orientation_q, quaternion_n2o
        )  # new -> original -> camera

        return remapped_orientation_q, remapped_extents

    def _get_o2n_object_rotation_matrix(self) -> np.ndarray:
        """Compute rotation matrix which rotates original to new object coordinates."""
        rotation_o2n = np.zeros((3, 3))  # original to new object convention
        if self._remap_y_axis == "x":
            rotation_o2n[0, 1] = 1
        elif self._remap_y_axis == "-x":
            rotation_o2n[0, 1] = -1
        elif self._remap_y_axis == "y":
            rotation_o2n[1, 1] = 1
        elif self._remap_y_axis == "-y":
            rotation_o2n[1, 1] = -1
        elif self._remap_y_axis == "z":
            rotation_o2n[2, 1] = 1
        elif self._remap_y_axis == "-z":
            rotation_o2n[2, 1] = -1
        else:
            raise ValueError("Unsupported remap_y_axis {self.remap_y}")

        if self._remap_x_axis == "x":
            rotation_o2n[0, 0] = 1
        elif self._remap_x_axis == "-x":
            rotation_o2n[0, 0] = -1
        elif self._remap_x_axis == "y":
            rotation_o2n[1, 0] = 1
        elif self._remap_x_axis == "-y":
            rotation_o2n[1, 0] = -1
        elif self._remap_x_axis == "z":
            rotation_o2n[2, 0] = 1
        elif self._remap_x_axis == "-z":
            rotation_o2n[2, 0] = -1
        else:
            raise ValueError("Unsupported remap_x_axis {self.remap_y}")

        # infer last column
        rotation_o2n[:, 2] = 1 - np.abs(np.sum(rotation_o2n, 1))  # rows must sum to +-1
        rotation_o2n[:, 2] *= np.linalg.det(rotation_o2n)  # make special orthogonal
        if np.linalg.det(rotation_o2n) != 1.0:  # check if special orthogonal
            raise ValueError("Unsupported combination of remap_{y,x}_axis. det != 1")
        return rotation_o2n

    def _quat_to_orientation_repr(self, quaternion: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to selected orientation representation.

        Args:
            quaternion:
                The quaternion to convert, scalar-last, shape (4,).

        Returns:
            The same orientation as represented by the quaternion in the chosen
            orientation representation.
        """
        if self._orientation_repr == "quaternion":
            return quaternion
        elif self._orientation_repr == "discretized":
            index = self._orientation_grid.quat_to_index(quaternion.numpy())
            return torch.tensor(
                index,
                dtype=torch.long,
            )
        else:
            raise NotImplementedError(
                f"Orientation representation {self._orientation_repr} is not supported."
            )

    def load_mesh(self, object_path: str) -> o3d.geometry.TriangleMesh:
        """Load an object mesh and adjust its object frame convention."""
        mesh = o3d.io.read_triangle_mesh(object_path)
        if self._remap_y_axis is None and self._remap_x_axis is None:
            return mesh
        elif self._remap_y_axis is None or self._remap_x_axis is None:
            raise ValueError("Either both or none of remap_{y,x}_axis have to be None.")

        rotation_o2n = self._get_o2n_object_rotation_matrix()
        mesh.rotate(
            rotation_o2n,
            center=np.array([0.0, 0.0, 0.0])[:, None],
        )
        return mesh


class ObjectError(Exception):
    """Error if something with the mesh is wrong."""

    pass


def _draw_depth_geometry(
    posed_mesh: o3d.geometry.TriangleMesh, camera: camera_utils.Camera
) -> np.ndarray:
    """Render a posed mesh given a camera looking along z axis (OpenCV convention)."""
    # see http://www.open3d.org/docs/latest/tutorial/visualization/customized_visualization.html

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=camera.width, height=camera.height, visible=False)

    # Add mesh in correct position
    vis.add_geometry(posed_mesh, True)

    options = vis.get_render_option()
    options.mesh_show_back_face = True

    # Set camera at fixed position (i.e., at 0,0,0, looking along z axis)
    view_control = vis.get_view_control()
    o3d_cam = camera.get_o3d_pinhole_camera_parameters()
    o3d_cam.extrinsic = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    view_control.convert_from_pinhole_camera_parameters(o3d_cam, True)

    # Generate the depth image
    vis.poll_events()
    vis.update_renderer()
    depth = np.asarray(vis.capture_depth_float_buffer())

    return depth
