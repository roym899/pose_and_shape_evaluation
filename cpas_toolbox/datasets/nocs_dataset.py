"""Module providing dataset class for NOCS datasets (CAMERA / REAL)."""
import datetime
import imghdr
import json
from glob import glob
import os
import pickle
from shutil import copyfile
import time
from typing import TypedDict, Optional, List
import zipfile

from joblib import Parallel, delayed
from scipy.spatial.transform import Rotation
import numpy as np
import open3d as o3d
import pandas as pd
import torch
from PIL import Image
import yoco
from tqdm import tqdm

from cpas_toolbox import pointset_utils, quaternion_utils, camera_utils, utils
from cpas_toolbox.datasets import nocs_utils


class NOCSDataset(torch.utils.data.Dataset):
    """Dataset class for NOCS dataset.

    CAMERA* and REAL* are training sets.
    CAMERA25 and REAL275 are test data.
    Some papers use CAMERA25 as validation when benchmarking REAL275.

    Datasets can be found here:
    https://github.com/hughw19/NOCS_CVPR2019/tree/master

    Expected directory format:
        {root_dir}/real_train/...
        {root_dir}/real_test/...
        {root_dir}/gts/...
        {root_dir}/obj_models/...
        {root_dir}/camera_composed_depth/...
        {root_dir}/val/...
        {root_dir}/train/...
    Which is easily obtained by downloading all the provided files and extracting them
    into the same directory.

    Necessary preprocessing of this data is performed during first initialization per
    and is saved to
        {root_dir}/csap_toolbox/...
    """

    num_categories = 7
    category_id_to_str = {
        0: "unknown",
        1: "bottle",
        2: "bowl",
        3: "camera",
        4: "can",
        5: "laptop",
        6: "mug",
    }
    category_str_to_id = {v: k for k, v in category_id_to_str.items()}

    class Config(TypedDict, total=False):
        """Configuration dictionary for NOCSDataset.

        Attributes:
            root_dir: See NOCSDataset docstring.
            split:
                The dataset split. The following strings are supported:
                    "camera_train": 275000 images, synthetic objects + real background
                    "camera_val": 25000 images, synthetic objects + real background
                    "real_train": 4300 images in 7 scenes, real
                    "real_test": 2750 images in 6 scenes, real
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
                If not None, the NOCS y-axis will be mapped to the provided axis.
                Resulting coordinate system will always be right-handed.
                This is typically the up-axis.
                Note that NOCS object models are NOT aligned the same as ShapeNetV2.
                To get ShapeNetV2 alignment: y
                One of: "x", "y", "z", "-x", "-y", "-z"
            remap_x_axis:
                If not None, the original x-axis will be mapped to the provided axis.
                Resulting coordinate system will always be right-handed.
                Note that NOCS object models are NOT aligned the same as ShapeNetV2.
                To get ShapeNetV2 alignment: -z
                One of: "x", "y", "z", "-x", "-y", "-z"
            category_str:
                If not None, only samples from the matching category will be returned.
                See NOCSDataset.category_id_to_str for admissible category strings.
        """

        root_dir: str
        split: str
        mask_pointcloud: bool
        normalize_pointcloud: bool
        scale_convention: str
        camera_convention: str
        orientation_repr: str
        remap_y_axis: Optional[str]
        remap_x_axis: Optional[str]
        category_str: Optional[str]

    default_config: Config = {
        "root_dir": None,
        "split": None,
        "mask_pointcloud": False,
        "normalize_pointcloud": False,
        "camera_convention": "opengl",
        "scale_convention": "half_max",
        "orientation_repr": "quaternion",
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
                with default_dict. See NOCSDataset.Config for supported keys.
        """
        config = yoco.load_config(config, current_dict=NOCSDataset.default_config)
        self._root_dir = utils.resolve_path(config["root_dir"])
        self._split = config["split"]
        self._check_dirs()
        self._camera_convention = config["camera_convention"]
        self._camera = self._get_split_camera()
        self._preprocess_path = os.path.join(
            self._root_dir, "csap_toolbox", self._split
        )
        if not os.path.isdir(self._preprocess_path):
            self._preprocess_dataset()
        self._mask_pointcloud = config["mask_pointcloud"]
        self._normalize_pointcloud = config["normalize_pointcloud"]
        self._scale_convention = config["scale_convention"]
        self._sample_files = self._get_sample_files(config["category_str"])
        self._remap_y_axis = config["remap_y_axis"]
        self._remap_x_axis = config["remap_x_axis"]
        self._orientation_repr = config["orientation_repr"]

    def _check_dirs(self) -> None:
        directories = [
            os.path.join(self._root_dir, "gts"),
        ]
        # required folders
        if all(os.path.exists(directory) for directory in directories):
            pass
        else:
            print(
                f"NOCS dataset ({self._split} split) not found, do you want to download"
                " it into the following directory:"
            )
            print("  ", self._root_dir)
            while True:
                decision = input("(Y/n) ").lower()
                if decision == "" or decision == "y":
                    self._download_dataset()
                    break
                elif decision == "n":
                    print("Dataset not found. Aborting.")
                    exit(0)

    def _get_dirs(self) -> List[str]:
        """Return required list of directories for current split."""
        dirs = []

        # gts only available for real test and camera val
        if self._split in ["real_test", "camera_val"]:
            dirs.append(os.path.join(self._root_dir, "gts"))

        dirs.append(os.path.join(self._root_dir, "obj_models"))

        # full depths for CAMERA
        if self._split in ["camera_val", "camera_train"]:
            dirs.append(os.path.join(self._root_dir, "camera_full_depths"))

        if self._split == "camera_train":
            dirs.append(os.path.join(self._root_dir, "train"))
        elif self._split == "camera_val":
            dirs.append(os.path.join(self._root_dir, "val"))
        elif self._split in ["real_train", "real_test"]:
            dirs.append(os.path.join(self._root_dir, self._split))
        else:
            raise ValueError(f"Specified split {self._split} is not supported.")

        return dirs

    def _download_dataset(self) -> None:
        missing_dirs = [d for d in self._get_dirs() if not os.path.exists(d)]
        for missing_dir in missing_dirs:
            download_dir, identifier = os.path.split(missing_dir)
            os.makedirs(download_dir, exist_ok=True)
            if identifier == "gts":
                zip_path = os.path.join(download_dir, "gts.zip")
                utils.download(
                    "http://download.cs.stanford.edu/orion/nocs/gts.zip", zip_path
                )
            elif identifier == "obj_models":
                zip_path = os.path.join(download_dir, "obj_models.zip")
                utils.download(
                    "http://download.cs.stanford.edu/orion/nocs/obj_models.zip",
                    zip_path,
                )
            elif identifier == "camera_full_depths":
                zip_path = os.path.join(download_dir, "camera_full_depths.zip")
                utils.download(
                    "http://download.cs.stanford.edu/orion/nocs/camera_composed_depth.zip",
                    zip_path,
                )
                z = zipfile.ZipFile(zip_path)
            elif identifier == "train":
                zip_path = os.path.join(download_dir, "train.zip")
                utils.download(
                    "http://download.cs.stanford.edu/orion/nocs/camera_train.zip",
                    zip_path,
                )
            elif identifier == "val":
                zip_path = os.path.join(download_dir, "val.zip")
                utils.download(
                    "http://download.cs.stanford.edu/orion/nocs/camera_val25K.zip",
                    zip_path,
                )
            elif identifier == "real_train":
                zip_path = os.path.join(download_dir, "real_train.zip")
                utils.download(
                    "http://download.cs.stanford.edu/orion/nocs/real_train.zip",
                    zip_path,
                )
            elif identifier == "real_test":
                zip_path = os.path.join(download_dir, "real_test.zip")
                utils.download(
                    "http://download.cs.stanford.edu/orion/nocs/real_test.zip", zip_path
                )
            else:
                raise ValueError(f"Downloading dir {missing_dir} unsupported.")
            z = zipfile.ZipFile(zip_path)
            z.extractall(download_dir)
            z.close()
            os.remove(zip_path)

    def __len__(self) -> int:
        """Return number of sample in dataset."""
        return len(self._sample_files)

    def __getitem__(self, idx: int) -> dict:
        """Return a sample of the dataset.

        Args:
            idx: Index of the instance.

        Returns:
            Sample containing the following items:
                "color"
                "depth"
                "mask"
                "pointset"
                "position"
                "orientation"
                "quaternion"
                "scale"
                "color_path"
        """
        sample_file = self._sample_files[idx]
        sample_data = pickle.load(open(sample_file, "rb"))
        sample = self._sample_from_sample_data(sample_data)
        return sample

    def _preprocess_dataset(self) -> None:
        """Create preprocessing files for the current split.

        One file per sample, which currently means per valid object mask will be
        created.

        Preprocessing will be stored on disk to {root_dir}/csap_toolbox/...
        This function will not store the preprocessing, so it still has to be loaded
        afterwards.
        """
        os.makedirs(self._preprocess_path)

        self._fix_obj_models()

        self._start_time = time.time()
        self._color_paths = self._get_color_files()

        Parallel(n_jobs=-1)(
            (
                delayed(self._preprocess_color_path)(i, color_path)
                for i, color_path in enumerate(self._color_paths)
            )
        )

        # store dictionary to map category to files
        sample_files = self._get_sample_files()
        category_str_to_files = {
            category_str: [] for category_str in NOCSDataset.category_id_to_str.values()
        }
        for sample_file in tqdm(sample_files):
            sample_data = pickle.load(open(sample_file, "rb"))
            category_id = sample_data["category_id"]
            category_str = NOCSDataset.category_id_to_str[category_id]
            _, file_name = os.path.split(sample_file)
            category_str_to_files[category_str].append(file_name)

        category_json_path = os.path.join(self._preprocess_path, "categories.json")
        with open(category_json_path, "w") as f:
            json.dump(dict(category_str_to_files), f)

        print(f"Finished preprocessing for {self._split}.", end="\033[K\n")

    def _fix_obj_models(self) -> None:
        """Fix issues with fileextensions.

        Some png files have jpg extension. This function fixes these models.
        """
        glob_pattern = os.path.join(self._root_dir, "**", "*.jpg")
        files = glob(glob_pattern, recursive=True)
        for filepath in files:
            what = imghdr.what(filepath)
            if what == "png":
                print("Fixing: ", filepath)
                folder, problematic_filename = os.path.split(filepath)
                name, _ = problematic_filename.split(".")
                fixed_filename = f"fixed_{name}.png"
                fixed_filepath = os.path.join(folder, fixed_filename)

                mtl_filepath = os.path.join(folder, "model.mtl")
                bu_mtl_filepath = os.path.join(folder, "model.mtl.old")
                copyfile(mtl_filepath, bu_mtl_filepath)

                copyfile(filepath, fixed_filepath)

    def _update_preprocess_progress(self, image_id: int) -> None:
        current_time = time.time()
        duration = current_time - self._start_time
        imgs_per_sec = image_id / duration
        if image_id > 10:
            remaining_imgs = len(self._color_paths) - image_id
            remaining_secs = remaining_imgs / imgs_per_sec
            remaining_time_str = str(datetime.timedelta(seconds=round(remaining_secs)))
        else:
            remaining_time_str = "N/A"
        print(
            f"Preprocessing image: {image_id:>10} / {len(self._color_paths)}"
            f" {image_id / len(self._color_paths) * 100:>6.2f}%"  # progress percentage
            f" Remaining time: {remaining_time_str}"  # remaining time
            "\033[K",  # clear until end of line
            end="\r",  # overwrite previous
        )

    def _preprocess_color_path(self, image_id: int, color_path: str) -> None:
        counter = 0
        self._update_preprocess_progress(image_id)

        depth_path = self._depth_path_from_color_path(color_path)
        if not os.path.isfile(depth_path):
            print(f"Missing depth file {depth_path}. Skipping.", end="\033[K\n")
            return

        mask_path = self._mask_path_from_color_path(color_path)
        meta_path = self._meta_path_from_color_path(color_path)
        meta_data = pd.read_csv(
            meta_path, sep=" ", header=None, converters={2: lambda x: str(x)}
        )
        instances_mask = self._load_mask(mask_path)
        mask_ids = np.unique(instances_mask).tolist()
        gt_id = 0  # GT only contains valid objects of interests and is 0-indexed
        for mask_id in mask_ids:

            if mask_id == 255:  # 255 is background
                continue
            match = meta_data[meta_data.iloc[:, 0] == mask_id]
            if match.empty:
                print(
                    f"Warning: mask {mask_id} not found in {meta_path}", end="\033[K\n"
                )
            elif match.shape[0] != 1:
                print(
                    f"Warning: mask {mask_id} not unique in {meta_path}", end="\033[K\n"
                )

            meta_row = match.iloc[0]
            category_id = meta_row.iloc[1]
            if category_id == 0:  # unknown / distractor object
                continue

            try:
                (
                    position,
                    orientation_q,
                    extents,
                    nocs_transform,
                ) = self._get_pose_and_scale(color_path, mask_id, gt_id, meta_row)
            except nocs_utils.PoseEstimationError:
                print(
                    "Insufficient data for pose estimation. "
                    f"Skipping {color_path}:{mask_id}.",
                    end="\033[K\n",
                )
                continue
            except ObjectError:
                print(
                    "Insufficient object mesh for pose estimation. "
                    f"Skipping {color_path}:{mask_id}.",
                    end="\033[K\n",
                )
                continue

            obj_path = self._get_obj_path(meta_row)
            sample_info = {
                "color_path": color_path,
                "depth_path": depth_path,
                "mask_path": mask_path,
                "mask_id": mask_id,
                "category_id": category_id,
                "obj_path": obj_path,
                "nocs_transform": nocs_transform,
                "position": position,
                "orientation_q": orientation_q,
                "extents": extents,
                "nocs_scale": torch.linalg.norm(extents),
                "max_extent": torch.max(extents),
            }
            out_file = os.path.join(
                self._preprocess_path, f"{image_id:08}_{counter}.pkl"
            )
            pickle.dump(sample_info, open(out_file, "wb"))
            counter += 1
            gt_id += 1

    def _get_color_files(self) -> list:
        """Return list of paths of color images of the selected split."""
        if self._split == "camera_train":
            glob_pattern = os.path.join(self._root_dir, "train", "**", "*_color.png")
            return sorted(glob(glob_pattern, recursive=True))
        elif self._split == "camera_val":
            glob_pattern = os.path.join(self._root_dir, "val", "**", "*_color.png")
            return sorted(glob(glob_pattern, recursive=True))
        elif self._split == "real_train":
            glob_pattern = os.path.join(
                self._root_dir, "real_train", "**", "*_color.png"
            )
            return sorted(glob(glob_pattern, recursive=True))
        elif self._split == "real_test":
            glob_pattern = os.path.join(
                self._root_dir, "real_test", "**", "*_color.png"
            )
            return sorted(glob(glob_pattern, recursive=True))
        else:
            raise ValueError(f"Specified split {self._split} is not supported.")

    def _get_sample_files(self, category_str: Optional[str] = None) -> list:
        """Return sorted list of sample file paths.

        Sample files are generated by NOCSDataset._preprocess_dataset.

        Args:
            category_str:
                If not None, only instances of the provided category will be returned.

        Returns:
            List of sample_data files.
        """
        glob_pattern = os.path.join(self._preprocess_path, "*.pkl")
        sample_files = glob(glob_pattern)
        sample_files.sort()
        if category_str is None:
            return sample_files
        if category_str not in NOCSDataset.category_str_to_id:
            raise ValueError(f"Unsupported category_str {category_str}.")

        category_json_path = os.path.join(self._preprocess_path, "categories.json")
        with open(category_json_path, "r") as f:
            category_str_to_filenames = json.load(f)
        filtered_sample_files = [
            os.path.join(self._preprocess_path, fn)
            for fn in category_str_to_filenames[category_str]
        ]
        return filtered_sample_files

    def _get_split_camera(self) -> None:
        """Return camera information for selected split."""
        # from: https://github.com/hughw19/NOCS_CVPR2019/blob/master/detect_eval.py
        if self._split in ["real_train", "real_test"]:
            return camera_utils.Camera(
                width=640,
                height=480,
                fx=591.0125,
                fy=590.16775,
                cx=322.525,
                cy=244.11084,
                pixel_center=0.0,
            )
        elif self._split in ["camera_train", "camera_val"]:
            return camera_utils.Camera(
                width=640,
                height=480,
                fx=577.5,
                fy=577.5,
                cx=319.5,
                cy=239.5,
                pixel_center=0.0,
            )
        else:
            raise ValueError(f"Specified split {self._split} is not supported.")

    def _sample_from_sample_data(self, sample_data: dict) -> dict:
        """Create dictionary containing a single sample."""
        color = torch.from_numpy(
            np.asarray(Image.open(sample_data["color_path"]), dtype=np.float32) / 255
        )
        depth = self._load_depth(sample_data["depth_path"])
        instances_mask = self._load_mask(sample_data["mask_path"])
        instance_mask = instances_mask == sample_data["mask_id"]

        pointcloud_mask = instance_mask if self._mask_pointcloud else None
        pointcloud = pointset_utils.depth_to_pointcloud(
            depth,
            self._camera,
            mask=pointcloud_mask,
            convention=self._camera_convention,
        )

        # adjust camera convention for position, orientation and scale
        position = pointset_utils.change_position_camera_convention(
            sample_data["position"], "opencv", self._camera_convention
        )

        # orientation / scale
        orientation_q, extents = self._change_axis_convention(
            sample_data["orientation_q"], sample_data["extents"]
        )
        orientation_q = pointset_utils.change_orientation_camera_convention(
            orientation_q, "opencv", self._camera_convention
        )
        orientation = self._quat_to_orientation_repr(orientation_q)
        scale = self._get_scale(sample_data, extents)

        # normalize pointcloud & position
        if self._normalize_pointcloud:
            pointcloud, centroid = pointset_utils.normalize_points(pointcloud)
            position = position - centroid

        sample = {
            "color": color,
            "depth": depth,
            "pointset": pointcloud,
            "mask": instance_mask,
            "position": position,
            "orientation": orientation,
            "quaternion": orientation_q,
            "scale": scale,
            "color_path": sample_data["color_path"],
            "obj_path": sample_data["obj_path"],
            "category_id": sample_data["category_id"],
            "category_str": NOCSDataset.category_id_to_str[sample_data["category_id"]],
        }
        return sample

    def _depth_path_from_color_path(self, color_path: str) -> str:
        """Return path to depth file from color filepath."""
        if self._split in ["real_train", "real_test"]:
            depth_path = color_path.replace("color", "depth")
        elif self._split in ["camera_train"]:
            depth_path = color_path.replace("color", "composed")
            depth_path = depth_path.replace("/train/", "/camera_full_depths/train/")
        elif self._split in ["camera_val"]:
            depth_path = color_path.replace("color", "composed")
            depth_path = depth_path.replace("/val/", "/camera_full_depths/val/")
        else:
            raise ValueError(f"Specified split {self._split} is not supported.")
        return depth_path

    def _mask_path_from_color_path(self, color_path: str) -> str:
        """Return path to mask file from color filepath."""
        mask_path = color_path.replace("color", "mask")
        return mask_path

    def _meta_path_from_color_path(self, color_path: str) -> str:
        """Return path to meta file from color filepath."""
        meta_path = color_path.replace("color.png", "meta.txt")
        return meta_path

    def _nocs_map_path_from_color_path(self, color_path: str) -> str:
        """Return NOCS map filepath from color filepath."""
        nocs_map_path = color_path.replace("color.png", "coord.png")
        return nocs_map_path

    def _get_pose_and_scale(
        self, color_path: str, mask_id: int, gt_id: int, meta_row: pd.Series
    ) -> tuple:
        """Return position, orientation, scale and NOCS transform.

        All of those follow OpenCV (x right, y down, z forward) convention.

        Args:
            color_path: Path to the color file.
            mask_id: Instance id in the instances mask.
            gt_id:
                Ground truth id. This is 0-indexed id of valid instances in meta file.
            meta_row:
                Matching row of meta file. Contains necessary information about mesh.

        Returns:
            position (torch.Tensor):
                Position of object center in camera frame. Shape (3,).
            quaternion (torch.Tensor):
                Orientation of object in camera frame.
                Scalar-last quaternion, shape (4,).
            extents (torch.Tensor):
                Bounding box side lengths.
            nocs_transformation (torch.Tensor):
                Transformation from centered [-0.5,0.5]^3 NOCS coordinates to camera.
        """
        gts_path = self._get_gts_path(color_path)
        obj_path = self._get_obj_path(meta_row)
        if self._split == "real_test":
            # only use gt for real test data, since there are errors in camera val
            gts_data = pickle.load(open(gts_path, "rb"))
            nocs_transform = gts_data["gt_RTs"][gt_id]
            position = nocs_transform[0:3, 3]
            rot_scale = nocs_transform[0:3, 0:3]
            nocs_scales = np.sqrt(np.sum(rot_scale ** 2, axis=0))
            rotation_matrix = rot_scale / nocs_scales[:, None]
            nocs_scale = nocs_scales[0]
        else:  # camera_train, camera_val, real_train
            # use ground truth NOCS mask to perform alignment
            (
                position,
                rotation_matrix,
                nocs_scale,
                nocs_transform,
            ) = self._estimate_object(color_path, mask_id)

        orientation_q = Rotation.from_matrix(rotation_matrix).as_quat()
        mesh_extents = self._get_mesh_extents_from_obj(obj_path)

        if "camera" in self._split:
            # CAMERA / ShapeNet meshes are normalized s.t. diagonal == 1
            # get metric extents by scaling with the diagonal
            extents = nocs_scale * mesh_extents
        elif "real" in self._split:
            # REAL object meshes are not normalized
            extents = mesh_extents
        else:
            raise ValueError(f"Specified split {self._split} is not supported.")

        position = torch.Tensor(position)
        orientation_q = torch.Tensor(orientation_q)
        extents = torch.Tensor(extents)
        nocs_transform = torch.Tensor(nocs_transform)
        return position, orientation_q, extents, nocs_transform

    def _get_gts_path(self, color_path: str) -> Optional[str]:
        """Return path to gts file from color filepath.

        Return None if split does not have ground truth information.
        """
        if self._split == "real_test":
            gts_folder = os.path.join(self._root_dir, "gts", "real_test")
        elif self._split == "camera_val":
            gts_folder = os.path.join(self._root_dir, "gts", "val")
        else:
            return None

        path = os.path.normpath(color_path)
        split_path = path.split(os.sep)
        number = path[-14:-10]
        gts_filename = f"results_{split_path[-3]}_{split_path[-2]}_{number}.pkl"
        gts_path = os.path.join(gts_folder, gts_filename)
        return gts_path

    def _get_obj_path(self, meta_row: pd.Series) -> str:
        """Return path to object file from meta data row."""
        if "camera" in self._split:  # ShapeNet mesh
            synset_id = meta_row.iloc[2]
            object_id = meta_row.iloc[3]
            obj_path = os.path.join(
                self._root_dir,
                "obj_models",
                self._split.replace("camera_", ""),
                synset_id,
                object_id,
                "model.obj",
            )
        elif "real" in self._split:  # REAL mesh
            object_id = meta_row.iloc[2]
            obj_path = os.path.join(
                self._root_dir, "obj_models", self._split, object_id + ".obj"
            )
        else:
            raise ValueError(f"Specified split {self._split} is not supported.")
        return obj_path

    def _get_mesh_extents_from_obj(self, obj_path: str) -> torch.Tensor:
        """Return maximum extent of bounding box from obj filepath.

        Note that this is normalized extent (with diagonal == 1) in the case of CAMERA
        dataset, and unnormalized (i.e., metric) extent in the case of REAL dataset.
        """
        mesh = o3d.io.read_triangle_mesh(obj_path)
        vertices = np.asarray(mesh.vertices)
        if len(vertices) == 0:
            raise ObjectError()
        extents = np.max(vertices, axis=0) - np.min(vertices, axis=0)
        return torch.Tensor(extents)

    def _load_mask(self, mask_path: str) -> torch.Tensor:
        """Load mask from mask filepath."""
        mask_img = np.asarray(Image.open(mask_path), dtype=np.uint8)
        if mask_img.ndim == 3 and mask_img.shape[2] == 4:  # CAMERA masks are RGBA
            instances_mask = mask_img[:, :, 0]  # use first channel only
        else:  # REAL masks are grayscale
            instances_mask = mask_img
        return torch.from_numpy(instances_mask)

    def _load_depth(self, depth_path: str) -> torch.Tensor:
        """Load depth from depth filepath."""
        depth = torch.from_numpy(
            np.asarray(Image.open(depth_path), dtype=np.float32) * 0.001
        )
        return depth

    def _load_nocs_map(self, nocs_map_path: str) -> torch.Tensor:
        """Load NOCS map from NOCS map filepath.

        Returns:
            NOCS map where each channel corresponds to one dimension in NOCS.
            Coordinates are normalized to [0,1], shape (H,W,3).
        """
        nocs_map = torch.from_numpy(
            np.asarray(Image.open(nocs_map_path), dtype=np.float32) / 255
        )
        # z-coordinate has to be flipped
        # see https://github.com/hughw19/NOCS_CVPR2019/blob/14dbce775c3c7c45bb7b19269bd53d68efb8f73f/dataset.py#L327 # noqa: E501
        nocs_map[:, :, 2] = 1 - nocs_map[:, :, 2]
        return nocs_map[:, :, :3]

    def _estimate_object(self, color_path: str, mask_id: int) -> tuple:
        """Estimate pose and scale through ground truth NOCS map."""
        position = rotation_matrix = scale = out_transform = None
        depth_path = self._depth_path_from_color_path(color_path)
        depth = self._load_depth(depth_path)
        mask_path = self._mask_path_from_color_path(color_path)
        instances_mask = self._load_mask(mask_path)
        instance_mask = instances_mask == mask_id
        nocs_map_path = self._nocs_map_path_from_color_path(color_path)
        nocs_map = self._load_nocs_map(nocs_map_path)
        valid_instance_mask = instance_mask * depth != 0
        nocs_map[~valid_instance_mask] = 0
        centered_nocs_points = nocs_map[valid_instance_mask] - 0.5

        measured_points = pointset_utils.depth_to_pointcloud(
            depth, self._camera, mask=valid_instance_mask, convention="opencv"
        )

        # require at least 30 point correspondences to prevent outliers
        if len(measured_points) < 30:
            raise nocs_utils.PoseEstimationError()

        # skip object if it cointains errorneous depth
        if torch.max(depth[valid_instance_mask]) > 32.0:
            print("Erroneous depth detected.", end="\033[K\n")
            raise nocs_utils.PoseEstimationError()

        (
            position,
            rotation_matrix,
            scale,
            out_transform,
        ) = nocs_utils.estimate_similarity_transform(
            centered_nocs_points, measured_points, verbose=False
        )

        if position is None:
            raise nocs_utils.PoseEstimationError()

        return position, rotation_matrix, scale, out_transform

    def _get_scale(self, sample_data: dict, extents: torch.Tensor) -> float:
        """Return scale from stored sample data and extents."""
        if self._scale_convention == "diagonal":
            return sample_data["nocs_scale"]
        elif self._scale_convention == "max":
            return sample_data["max_extent"]
        elif self._scale_convention == "half_max":
            return 0.5 * sample_data["max_extent"]
        elif self._scale_convention == "full":
            return extents
        else:
            raise ValueError(
                f"Specified scale convention {self._scale_convnetion} not supported."
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
