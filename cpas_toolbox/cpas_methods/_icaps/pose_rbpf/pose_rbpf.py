import copy
import os

import numpy as np
import torch
from transforms3d import euler, quaternions

from ..deep_sdf import deepsdf_optim
from ..deep_sdf.evaluator import Evaluator
from ..models import aae_models, pointnet2_msg
from ..pointnet2 import pointnet2_utils
from ..utils import decoder_utils, deepsdf_utils, poserbpf_utils
from . import particle_filter

NPOINTS = 4873
NP_THRESHOLD = 0

class PoseRBPF:
    def __init__(
        self,
        obj_list: list,
        cfg_list: list,
        ckpt_dir: str,
        deepsdf_ckp_folder: str,
        latentnet_ckp_folder: str,
        model_dir="../category-level_models/",
        test_model_dir="../obj_models/real_test/",
        visualize=True,
        device="cuda",
    ):
        """

        Args:
            ckpt_dir: Folder containing checkpoints.
        """

        self.visualize = visualize
        self.device = device

        self.obj_list = obj_list

        # load the object information
        self.cfg_list = cfg_list

        # load encoders and poses
        self.aae_list = []
        self.codebook_list = []
        self.rbpf_list = []
        self.rbpf_ok_list = []
        for obj in self.obj_list:
            ckpt_file = "{}/ckpt_{}_0300.pth".format(ckpt_dir, obj)
            codebook_file = "{}/codebook_{}_0300.pth".format(ckpt_dir, obj)
            self.aae_full = aae_models.AAE(
                [obj], capacity=1, code_dim=128, device=self.device
            )
            self.aae_full.encoder.eval()
            self.aae_full.decoder.eval()
            for param in self.aae_full.encoder.parameters():
                param.requires_grad = False
            for param in self.aae_full.decoder.parameters():
                param.requires_grad = False
            checkpoint = torch.load(ckpt_file, map_location=self.device)
            self.aae_full.load_ckpt_weights(checkpoint["aae_state_dict"])
            self.aae_list.append(copy.deepcopy(self.aae_full.encoder))
            if not os.path.exists(codebook_file):
                print("Cannot find codebook in : " + codebook_file)
                print(
                    "Codebook generation not supported in cpas_toolbox. "
                    "Refer to original repo."
                )

            self.codebook_list.append(
                torch.load(codebook_file, map_location=self.device)[0]
            )
            self.rbpf_codepose = (
                torch.load(codebook_file, map_location=self.device)[1].cpu().numpy()
            )  # all are identical
            idx_obj = self.obj_list.index(obj)
            self.rbpf_list.append(
                particle_filter.ParticleFilter(
                    self.cfg_list[idx_obj].PF,
                    n_particles=self.cfg_list[idx_obj].PF.N_PROCESS,
                    device=self.device,
                )
            )
            self.rbpf_ok_list.append(False)

        # renderer
        self.intrinsics = np.array(
            [
                [self.cfg_list[0].PF.FU, 0, self.cfg_list[0].PF.U0],
                [0, self.cfg_list[0].PF.FV, self.cfg_list[0].PF.V0],
                [0, 0, 1.0],
            ],
            dtype=np.float32,
        )

        # target object property
        self.target_obj = None
        self.target_obj_idx = None
        self.target_obj_encoder = None
        self.target_obj_codebook = None
        self.target_obj_cfg = None

        # initialize the particle filters
        self.rbpf = particle_filter.ParticleFilter(
            self.cfg_list[0].PF,
            n_particles=self.cfg_list[0].PF.N_PROCESS,
            device=self.device,
        )
        self.rbpf_ok = False

        # pose rbpf for initialization
        self.rbpf_init_max_sim = 0

        # data properties
        self.data_with_gt = False
        self.data_with_est_bbox = False
        self.data_with_est_center = False
        self.data_intrinsics = np.ones((3, 3), dtype=np.float32)

        # initialize the PoseRBPF variables
        # ground truth information
        self.gt_available = False
        self.gt_t = [0, 0, 0]
        self.gt_rotm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        self.gt_bbox_center = np.zeros((3,))
        self.gt_bbox_size = 0
        self.gt_uv = np.array([0, 0, 1], dtype=np.float32)
        self.gt_z = 0
        self.gt_scale = 1.0

        # estimated states
        self.est_bbox_center = np.zeros((2, self.cfg_list[0].PF.N_PROCESS))
        self.est_bbox_size = np.zeros((self.cfg_list[0].PF.N_PROCESS,))
        self.est_bbox_weights = np.zeros((self.cfg_list[0].PF.N_PROCESS,))

        # for logging
        self.log_err_rx = []
        self.log_err_ry = []
        self.log_err_rz = []
        self.log_err_r = []
        self.log_err_r_star = []
        self.log_max_sim = []
        self.log_dir = "./"
        self.log_created = False
        self.log_shape_created = False
        self.log_pose = None
        self.log_shape = None
        self.log_error = None
        self.log_err_uv = []
        # posecnn prior
        self.prior_uv = [0, 0, 1]
        self.prior_z = 0
        self.prior_t = np.array([0, 0, 0], dtype=np.float32)
        self.prior_R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

        # flags for experiments
        self.exp_with_mask = False
        self.step = 0
        self.iskf = False
        self.init_step = False
        self.save_uncertainty = False
        self.show_prior = False

        # motion model
        self.T_c1c0 = np.eye(4, dtype=np.float32)
        self.T_o0o1 = np.eye(4, dtype=np.float32)
        self.T_c0o = np.eye(4, dtype=np.float32)
        self.T_c1o = np.eye(4, dtype=np.float32)
        self.Tbr1 = np.eye(4, dtype=np.float32)
        self.Tbr0 = np.eye(4, dtype=np.float32)
        self.Trc = np.eye(4, dtype=np.float32)
        self.embeddings_prev = None

        # deepsdf
        experiment_directory = "{}/{}s/".format(deepsdf_ckp_folder, obj_list[0])
        self.decoder = decoder_utils.load_decoder(
            experiment_directory, 2000, device=self.device
        )
        self.decoder = self.decoder.module.to(self.device)
        self.evaluator = Evaluator(self.decoder)
        latent_size = 256
        std_ = 0.01
        self.rand_tensor = torch.ones(1, latent_size).normal_(mean=0, std=std_)
        self.latent_tensor = self.rand_tensor.float().to(self.device)
        self.latent_tensor.requires_grad = False
        self.mask = None
        self.sdf_optim = deepsdf_optim.DeepSDFOptimizer(
            self.decoder, optimize_shape=False, device=self.device
        )
        # test_model_dir = "../obj_models/real_test/"
        # fn = "{}{}_vertices.txt".format(test_model_dir, test_instance)
        # points = np.loadtxt(fn, dtype=np.float32)  # n x 3
        # self.size_gt = np.max(np.linalg.norm(points, axis=1))
        # self.points_gt = torch.from_numpy(points)
        self.latent_tensor_initialized = False
        # self.size_gt_pn = get_bbox_dist(points)  # metric ground truth diagonal
        # self.size_est = self.size_gt_pn
        # self.ratio = self.size_gt_pn / self.size_gt

        # points_obj_norm = self.points_gt
        # Transform from Nocs object frame to ShapeNet object frame
        # rotm_obj2shapenet = euler.euler2mat(0.0, np.pi / 2.0, 0.0)
        # points_obj_shapenet = np.dot(rotm_obj2shapenet, points_obj_norm.T).T
        # points_obj_shapenet = np.float32(points_obj_shapenet)
        # self.points_c = torch.from_numpy(points_obj_shapenet)

        # partial point cloud observation
        self.points_o_partial = None

        # pointnet++
        ckpt_file_pn = "{}/{}/ckpt_{}_0300.pth".format(
            latentnet_ckp_folder, self.obj_list[0], self.obj_list[0]
        )
        # self.label_gt_path = "./latent_gt/{}_sfs.pth".format(test_instance)
        pn_checkpoint = torch.load(ckpt_file_pn, map_location=self.device)
        self.model = pointnet2_msg.get_model(input_channels=0)
        self.model.load_ckpt_weights(pn_checkpoint["aae_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        # self.label_gt = torch.load(self.label_gt_path)[0, :, :].detach()
        # self.label_gt.requires_grad = False
        self.loss_shape_refine = 10000
        self.loss_last = 10000
        self.dist_init = 0
        self.dist_opt = 0

        # for debugging
        self.T_filter = None
        self.T_refine = None
        self.latent_vec_pointnet = None
        self.latent_vec_refine = None
        self.points_o_est_vis = None
        self.points_o_refine_vis = None
        # self.points_gt_aug = torch.ones(
        #     self.points_gt.shape[0], self.points_gt.shape[1] + 1
        # )
        # self.points_gt_aug[:, :3] = self.points_gt
        # self.points_gt_aug = self.points_gt_aug.to(self.device)
        self.fps = []

    def set_ratio(self, ratio):
        self.ratio = ratio
        self.sdf_optim.ratio = self.ratio * 1.0

    def reset(self):
        self.rbpf_list = []
        self.rbpf_ok_list = []
        for obj in self.obj_list:
            idx_obj = self.obj_list.index(obj)
            self.rbpf_list.append(
                particle_filter.ParticleFilter(
                    self.cfg_list[idx_obj].PF,
                    n_particles=self.cfg_list[idx_obj].PF.N_PROCESS,
                    device=self.device,
                )
            )
            self.rbpf_ok_list.append(False)

        # renderer
        self.intrinsics = np.array(
            [
                [self.cfg_list[0].PF.FU, 0, self.cfg_list[0].PF.U0],
                [0, self.cfg_list[0].PF.FV, self.cfg_list[0].PF.V0],
                [0, 0, 1.0],
            ],
            dtype=np.float32,
        )

        # target object property
        self.target_obj = None
        self.target_obj_idx = None
        self.target_obj_encoder = None
        self.target_obj_codebook = None
        self.target_obj_cfg = None

        # initialize the particle filters
        self.rbpf = particle_filter.ParticleFilter(
            self.cfg_list[0].PF,
            n_particles=self.cfg_list[0].PF.N_PROCESS,
            device=self.device,
        )
        self.rbpf_ok = False

        # pose rbpf for initialization
        self.rbpf_init_max_sim = 0

        # data properties
        self.data_with_gt = False
        self.data_with_est_bbox = False
        self.data_with_est_center = False
        self.data_intrinsics = np.ones((3, 3), dtype=np.float32)

        # initialize the PoseRBPF variables
        # ground truth information
        self.gt_available = False
        self.gt_t = [0, 0, 0]
        self.gt_rotm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        self.gt_bbox_center = np.zeros((3,))
        self.gt_bbox_size = 0
        self.gt_uv = np.array([0, 0, 1], dtype=np.float32)
        self.gt_z = 0
        self.gt_scale = 1.0

        # estimated states
        self.est_bbox_center = np.zeros((2, self.cfg_list[0].PF.N_PROCESS))
        self.est_bbox_size = np.zeros((self.cfg_list[0].PF.N_PROCESS,))
        self.est_bbox_weights = np.zeros((self.cfg_list[0].PF.N_PROCESS,))

        # for logging
        self.log_err_rx = []
        self.log_err_ry = []
        self.log_err_rz = []
        self.log_err_r = []
        self.log_err_r_star = []
        self.log_max_sim = []
        self.log_dir = "./"
        self.log_created = False
        self.log_shape_created = False
        self.log_pose = None
        self.log_shape = None
        self.log_error = None
        self.log_err_uv = []
        # posecnn prior
        self.prior_uv = [0, 0, 1]
        self.prior_z = 0
        self.prior_t = np.array([0, 0, 0], dtype=np.float32)
        self.prior_R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

        # flags for experiments
        self.exp_with_mask = False
        self.step = 0
        self.iskf = False
        self.init_step = False
        self.save_uncertainty = False
        self.show_prior = False

        # motion model
        self.T_c1c0 = np.eye(4, dtype=np.float32)
        self.T_o0o1 = np.eye(4, dtype=np.float32)
        self.T_c0o = np.eye(4, dtype=np.float32)
        self.T_c1o = np.eye(4, dtype=np.float32)
        self.Tbr1 = np.eye(4, dtype=np.float32)
        self.Tbr0 = np.eye(4, dtype=np.float32)
        self.Trc = np.eye(4, dtype=np.float32)
        self.embeddings_prev = None

        # deepsdf
        latent_size = 256
        std_ = 0.01
        self.rand_tensor = torch.ones(1, latent_size).normal_(mean=0, std=std_)
        self.latent_tensor = self.rand_tensor.float().to(self.device)
        self.latent_tensor.requires_grad = False
        self.mask = None
        self.sdf_optim = deepsdf_optim.DeepSDFOptimizer(
            self.decoder, optimize_shape=False, device=self.device
        )
        # test_model_dir = "../obj_models/real_test/"
        # fn = "{}{}_vertices.txt".format(test_model_dir, test_instance)
        # points = np.loadtxt(fn, dtype=np.float32)  # n x 3
        # self.size_gt = np.max(np.linalg.norm(points, axis=1))
        # self.points_gt = torch.from_numpy(points)
        self.latent_tensor_initialized = False
        # self.size_gt_pn = get_bbox_dist(points)  # metric ground truth diagonal
        # self.size_est = self.size_gt_pn
        # self.ratio = self.size_gt_pn / self.size_gt
        self.ratio = self.ratio
        self.sdf_optim.ratio = self.ratio * 1.0

        # points_obj_norm = self.points_gt
        # Transform from Nocs object frame to ShapeNet object frame
        # rotm_obj2shapenet = euler.euler2mat(0.0, np.pi / 2.0, 0.0)
        # points_obj_shapenet = np.dot(rotm_obj2shapenet, points_obj_norm.T).T
        # points_obj_shapenet = np.float32(points_obj_shapenet)
        # self.points_c = torch.from_numpy(points_obj_shapenet)

        # partial point cloud observation
        self.points_o_partial = None

        self.loss_shape_refine = 10000
        self.loss_last = 10000
        self.dist_init = 0
        self.dist_opt = 0

        # for debugging
        self.T_filter = None
        self.T_refine = None
        self.latent_vec_pointnet = None
        self.latent_vec_refine = None
        self.points_o_est_vis = None
        self.points_o_refine_vis = None
        # self.points_gt_aug = torch.ones(
        #     self.points_gt.shape[0], self.points_gt.shape[1] + 1
        # )
        # self.points_gt_aug[:, :3] = self.points_gt
        # self.points_gt_aug = self.points_gt_aug.to(self.device)
        self.fps = []

    # specify the target object for tracking
    def set_target_obj(self, target_object):
        assert (
            target_object in self.obj_list
        ), "target object {} is not in the list of test objects".format(target_object)
        # set target object property
        self.target_obj = target_object
        self.target_obj_idx = self.obj_list.index(target_object)
        self.target_obj_encoder = self.aae_list[self.target_obj_idx]
        self.target_obj_codebook = self.codebook_list[self.target_obj_idx]
        self.target_obj_cfg = self.cfg_list[self.target_obj_idx]

        self.target_obj_cfg.PF.FU = self.intrinsics[0, 0]
        self.target_obj_cfg.PF.FV = self.intrinsics[1, 1]
        self.target_obj_cfg.PF.U0 = self.intrinsics[0, 2]
        self.target_obj_cfg.PF.V0 = self.intrinsics[1, 2]

        # reset particle filter
        self.rbpf = self.rbpf_list[self.target_obj_idx]
        self.rbpf_ok = self.rbpf_ok_list[self.target_obj_idx]
        self.rbpf_init_max_sim = 0

        # estimated states
        self.est_bbox_center = np.zeros((2, self.target_obj_cfg.PF.N_PROCESS))
        self.est_bbox_size = np.zeros((self.target_obj_cfg.PF.N_PROCESS,))
        self.est_bbox_weights = np.zeros((self.target_obj_cfg.PF.N_PROCESS,))

        # for logging
        self.log_err_rx = []
        self.log_err_ry = []
        self.log_err_rz = []
        self.log_err_r = []
        self.log_err_r_star = []
        self.log_max_sim = []
        self.log_dir = "./"
        self.log_created = False
        self.log_shape_created = False
        self.log_pose = None
        self.log_shape = None
        self.log_error = None

        # posecnn prior
        self.prior_uv = [0, 0, 1]
        self.prior_z = 0
        self.prior_t = np.array([0, 0, 0], dtype=np.float32)
        self.prior_R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

        # motion model
        self.T_c1c0 = np.eye(4, dtype=np.float32)
        self.T_o0o1 = np.eye(4, dtype=np.float32)
        self.T_c0o = np.eye(4, dtype=np.float32)
        self.T_c1o = np.eye(4, dtype=np.float32)
        self.Tbr1 = np.eye(4, dtype=np.float32)
        self.Tbr0 = np.eye(4, dtype=np.float32)
        self.Trc = np.eye(4, dtype=np.float32)

    def use_detection_priors(self, n_particles):
        self.rbpf.uv[-n_particles:] = np.repeat([self.prior_uv], n_particles, axis=0)
        self.rbpf.uv[-n_particles:, :2] += np.random.uniform(
            -self.target_obj_cfg.PF.UV_NOISE_PRIOR,
            self.target_obj_cfg.PF.UV_NOISE_PRIOR,
            (n_particles, 2),
        )

    # initialize PoseRBPF
    def initialize_poserbpf(
        self, image, intrinsics, uv_init, n_init_samples, scale_prior, depth=None
    ):
        # sample around the center of bounding box
        uv_h = np.array([uv_init[0], uv_init[1], 1])
        uv_h = np.repeat(np.expand_dims(uv_h, axis=0), n_init_samples, axis=0)
        uv_h[:, :2] += np.random.uniform(
            -self.target_obj_cfg.PF.INIT_UV_NOISE,
            self.target_obj_cfg.PF.INIT_UV_NOISE,
            (n_init_samples, 2),
        )
        uv_h[:, 0] = np.clip(uv_h[:, 0], 0, image.shape[1])
        uv_h[:, 1] = np.clip(uv_h[:, 1], 0, image.shape[0])
        self.uv_init = uv_h.copy()

        uv_h_int = uv_h.astype(int)
        uv_h_int[:, 0] = np.clip(uv_h_int[:, 0], 0, image.shape[1] - 1)
        uv_h_int[:, 1] = np.clip(uv_h_int[:, 1], 0, image.shape[0] - 1)
        depth_np = depth.numpy()
        z = depth_np[uv_h_int[:, 1], uv_h_int[:, 0], 0]
        z = np.expand_dims(z, axis=1)
        z[z > 0] += np.random.uniform(-0.25, 0.05, z[z > 0].shape)
        z[z == 0] = np.random.uniform(0.5, 1.5, z[z == 0].shape)
        self.z_init = z.copy()

        scale_h = np.array([scale_prior])
        scale_h = np.repeat(np.expand_dims(scale_h, axis=0), n_init_samples, axis=0)
        scale_h += np.random.randn(n_init_samples, 1) * 0.05

        # evaluate translation
        distribution, _ = self.evaluate_particles(
            depth,
            uv_h,
            z,
            scale_h,
            self.target_obj_cfg.TRAIN.RENDER_DIST[0],
            0.1,
            depth,
            debug_mode=False,
        )

        # find the max pdf from the distribution matrix
        self.index_star = poserbpf_utils.my_arg_max(distribution)
        uv_star = uv_h[self.index_star[0], :]  # .copy()
        z_star = z[self.index_star[0], :]  # .copy()
        scale_star = scale_h[self.index_star[0], :]
        self.rbpf.update_trans_star_uvz(uv_star, z_star, scale_star, intrinsics)
        distribution[self.index_star[0], :] /= torch.sum(
            distribution[self.index_star[0], :]
        )
        self.rbpf.rot = (
            distribution[self.index_star[0], :]
            .view(1, 1, 37, 72, 72)
            .repeat(self.rbpf.n_particles, 1, 1, 1, 1)
        )

        self.rbpf.update_rot_star_R(
            quaternions.quat2mat(self.rbpf_codepose[self.index_star[1]][3:])
        )
        self.rbpf.rot_bar = self.rbpf.rot_star
        self.rbpf.trans_bar = self.rbpf.trans_star
        self.rbpf.uv_bar = uv_star
        self.rbpf.z_bar = z_star
        self.rbpf.scale_star = scale_star
        self.rbpf.scale_bar = scale_star
        self.rbpf_init_max_sim = self.log_max_sim[-1]

        # initialize shape latent vector
        depth_np = depth_np[:, :, 0] * self.mask
        depth_vis = depth_np * 1.0
        points_np = deepsdf_utils.depth2pc(
            depth_np, depth_np.shape[0], depth_np.shape[1], intrinsics
        )  # n x 4
        # points_c = torch.from_numpy(points_np).to(self.device)
        T_init = np.eye(4, dtype=np.float32)
        T_init[:3, :3] = self.rbpf.rot_bar
        T_init[:3, 3] = self.rbpf.trans_bar
        self.size_est = self.rbpf.scale_bar[0]

        if len(points_np) > 0:
            self.sdf_optim.size_est = torch.tensor(self.size_est / self.ratio).to(
                self.device
            )
            # with this
            points_choice = self.sample_points(points_np[:, :3])
            self.latent_vector_prediction_pn(
                points_choice, self.rbpf.rot_bar, self.rbpf.trans_bar
            )
            self.latent_vec_optim = self.latent_tensor.clone()
        else:
            # print(
            #     "*** NOTHING ON THE DEPTH IMAGE! ... USING ORIGINAL BACKPROGATION METHOD ***"
            # )
            # self.initialize_latent_vector(points_c, T_init)  # this doesn't work ?

            print("*** NOTHING ON THE DEPTH IMAGE! ... USING RANDOM LATENT ***")
            # taken from __init__
            latent_size = 256
            std_ = 0.01
            self.rand_tensor = torch.ones(1, latent_size).normal_(mean=0, std=std_)
            self.latent_tensor = self.rand_tensor.float().to(self.device)
            self.latent_vec_optim = self.latent_tensor.clone()

    def latent_vector_prediction_pn(self, points_choice, rot, trans):

        # Transform from camera frame to object frame
        points_obj = np.dot(
            rot.T, (points_choice.T - np.tile(trans, (len(points_choice), 1)).T)
        ).T

        # Normalization
        points_obj_norm = points_obj / self.size_est

        # Transform from Nocs object frame to ShapeNet object frame
        rotm_obj2shapenet = euler.euler2mat(0.0, np.pi / 2.0, 0.0)
        points_obj_shapenet = np.dot(rotm_obj2shapenet, points_obj_norm.T).T
        points_obj_shapenet = np.float32(points_obj_shapenet)
        # visualize_depth_pc(points_obj_norm, points_obj_shapenet)
        self.points_o_partial = points_obj_shapenet * self.size_est

        points_c = torch.from_numpy(points_obj_shapenet).to(self.device).unsqueeze(0)
        pred_label = self.model(points_c)
        self.latent_tensor = pred_label.clone()

    # evaluate particles according to the RGB(D) images
    def evaluate_particles(
        self,
        image,
        uv,
        z,
        scale,
        render_dist,
        gaussian_std,
        depth,
        debug_mode=False,
        run_deep_sdf=False,
    ):

        # crop the rois from input depth image
        images_roi_cuda = poserbpf_utils.trans_zoom_uvz_cuda(
            depth.detach(),
            uv,
            z,
            scale,
            self.target_obj_cfg.PF.FU,
            self.target_obj_cfg.PF.FV,
            render_dist,
            device=self.device,
        ).detach()

        # normalize the depth
        n_particles = z.shape[0]
        z_cuda = torch.from_numpy(z).float().to(self.device).unsqueeze(2).unsqueeze(3)
        scale_cuda = (
            torch.from_numpy(scale).float().to(self.device).unsqueeze(2).unsqueeze(3)
        )
        images_roi_cuda = (images_roi_cuda - z_cuda) / scale_cuda + 0.5
        images_roi_cuda = torch.clamp(images_roi_cuda, 0, 1)

        # forward passing
        codes = (
            self.target_obj_encoder.forward(images_roi_cuda)
            .view(n_particles, -1)
            .detach()
        )
        self.inputs = images_roi_cuda
        # self.recon, self.code_rec = self.aae_full.forward(images_roi_cuda)

        # compute the similarity between particles' codes and the codebook
        cosine_distance_matrix = self.aae_full.compute_distance_matrix(
            codes, self.target_obj_codebook
        )

        # prior distance
        if self.embeddings_prev is not None:
            prior_dists = self.aae_full.compute_distance_matrix(
                codes, self.embeddings_prev
            )
            if torch.max(prior_dists) < 0.8:
                prior_dists = None
        else:
            prior_dists = None

        if prior_dists is not None:
            cosine_distance_matrix = cosine_distance_matrix + prior_dists.repeat(
                1, cosine_distance_matrix.size(1)
            )

        # get the maximum similarity for each particle
        v_sims, i_sims = torch.max(cosine_distance_matrix, dim=1)
        self.cos_dist_mat = v_sims

        # compute distribution from similarity
        max_sim_all, i_sim_all = torch.max(v_sims, dim=0)
        self.log_max_sim.append(max_sim_all)
        pdf_matrix = poserbpf_utils.mat2pdf(
            cosine_distance_matrix / max_sim_all, 1, gaussian_std
        )

        # evaluate with deepsdf
        sdf_scores = torch.from_numpy(np.ones_like(z)).to(self.device).float()

        if run_deep_sdf:
            distances = np.ones_like(z)
            depth_np = depth.numpy()[:, :, 0] * self.mask
            points_np = deepsdf_utils.depth2pc(
                depth_np, depth_np.shape[0], depth_np.shape[1], self.intrinsics
            )
            points_c = torch.from_numpy(points_np).to(self.device)
            if len(points_c) > NP_THRESHOLD:
                for i in range(self.rbpf.n_particles):
                    R = quaternions.quat2mat(self.rbpf_codepose[i_sims[i]][3:])
                    t = back_project(uv[i], self.intrinsics, z[i])
                    R = add_trans_R(t, R)
                    T_co = np.eye(4, dtype=np.float32)
                    T_co[:3, :3] = R
                    T_co[:3, 3] = t
                    distances[i] = self.sdf_optim.eval_latent_vector(
                        T_co, points_c, self.latent_tensor
                    )

                distances_temp = distances / np.min(distances)
                distances_torch = (
                    torch.from_numpy(distances_temp).to(self.device).float()
                )
                sdf_scores = mat2pdf(distances_torch, 1.0, 0.5)
                pdf_matrix = torch.mul(pdf_matrix, sdf_scores)

        return pdf_matrix, prior_dists

    # filtering
    def process_poserbpf(
        self,
        image,
        intrinsics,
        use_detection_prior=True,
        depth=None,
        debug_mode=False,
        run_deep_sdf=False,
    ):

        # propagation
        uv_noise = self.target_obj_cfg.PF.UV_NOISE
        z_noise = self.target_obj_cfg.PF.Z_NOISE
        self.rbpf.add_noise_r3(uv_noise, z_noise)
        self.rbpf.add_noise_rot()

        # poserbpf++
        if use_detection_prior and self.prior_uv[0] > 0 and self.prior_uv[1] > 0:
            self.use_detection_priors(int(self.rbpf.n_particles / 2))

        # compute pdf matrix for each particle
        est_pdf_matrix, prior_dists = self.evaluate_particles(
            depth,
            self.rbpf.uv,
            self.rbpf.z,
            self.rbpf.scale,
            self.target_obj_cfg.TRAIN.RENDER_DIST[0],
            self.target_obj_cfg.PF.WT_RESHAPE_VAR,
            depth,
            debug_mode=debug_mode,
            run_deep_sdf=run_deep_sdf,
        )

        # most likely particle
        temp_indext_star = poserbpf_utils.my_arg_max(est_pdf_matrix)
        if temp_indext_star is not None:
            self.index_star = temp_indext_star

        uv_star = self.rbpf.uv[self.index_star[0], :].copy()
        z_star = self.rbpf.z[self.index_star[0], :].copy()
        # scale_star = self.rbpf.scale[self.index_star[0], :].copy()
        self.rbpf.update_trans_star(uv_star, z_star, intrinsics)
        self.rbpf.update_rot_star_R(
            quaternions.quat2mat(self.rbpf_codepose[self.index_star[1]][3:])
        )

        # match rotation distribution
        self.rbpf.rot = torch.clamp(self.rbpf.rot, 1e-6, 1)
        rot_dist = torch.exp(
            torch.add(
                torch.log(est_pdf_matrix),
                torch.log(self.rbpf.rot.view(self.rbpf.n_particles, -1)),
            )
        )
        normalizers = torch.sum(rot_dist, dim=1)

        normalizers_cpu = normalizers.cpu().numpy()
        self.rbpf.weights = normalizers_cpu / np.sum(normalizers_cpu)

        if prior_dists is not None:
            self.rbpf.weights = self.rbpf.weights
            self.rbpf.weights /= np.sum(self.rbpf.weights)

        rot_dist /= normalizers.unsqueeze(1).repeat(1, self.target_obj_codebook.size(0))

        # matched distributions
        self.rbpf.rot = rot_dist.view(self.rbpf.n_particles, 1, 37, 72, 72)

        # resample
        self.rbpf.resample_ddpf(self.rbpf_codepose, intrinsics, self.target_obj_cfg.PF)
        self.size_est = self.rbpf.scale_bar[0]

        # compute previous embeddings
        images_roi_cuda = poserbpf_utils.trans_zoom_uvz_cuda(
            depth.detach(),
            np.expand_dims(self.rbpf.uv_bar, 0),
            np.expand_dims(self.rbpf.z_bar, 0),
            np.expand_dims(self.rbpf.scale_bar, 0),
            self.target_obj_cfg.PF.FU,
            self.target_obj_cfg.PF.FV,
            self.target_obj_cfg.TRAIN.RENDER_DIST[0],
            device=self.device,
        ).detach()

        # normalize the depth
        z = np.expand_dims(self.rbpf.z_bar, 0)
        z_cuda = torch.from_numpy(z).float().to(self.device).unsqueeze(2).unsqueeze(3)
        scale_cuda = (
            torch.from_numpy(np.expand_dims(self.rbpf.scale_bar, 0))
            .float()
            .to(self.device)
            .unsqueeze(2)
            .unsqueeze(3)
        )
        images_roi_cuda = (images_roi_cuda - z_cuda) / scale_cuda + 0.5
        images_roi_cuda = torch.clamp(images_roi_cuda, 0, 1)

        # forward passing
        codes = self.target_obj_encoder.forward(images_roi_cuda).view(1, -1).detach()
        self.embeddings_prev = codes

        return 0

    def refine_pose_and_shape(self, depth, intrinsics, refine_steps=50):

        # initialize shape latent vector
        depth_np = depth.numpy()[:, :, 0] * self.mask
        points_np = deepsdf_utils.depth2pc(
            depth_np, depth_np.shape[0], depth_np.shape[1], intrinsics.numpy()[0]
        )
        points_c = torch.from_numpy(points_np).to(self.device)

        if len(points_np) > NP_THRESHOLD:
            points_choice = self.sample_points(points_np[:, :3])
            self.latent_vector_prediction_pn(
                points_choice, self.rbpf.rot_bar, self.rbpf.trans_bar
            )
            # print("LatentNet inference time = ", time_elapse)
            if len(points_np) > NPOINTS:
                points_c = torch.from_numpy(
                    np.hstack(
                        (
                            points_choice,
                            np.ones((points_choice.shape[0], 1), dtype=np.float32),
                        )
                    )
                ).to(self.device)

        else:
            print("NOT ENOUGH POINTS ON THE DEPTH IMAGE! SKIP REFINEMENT")
            return
        # debug information
        self.latent_vec_pointnet = self.latent_tensor.clone().detach()

        T_co = np.eye(4, dtype=np.float32)
        T_co[:3, :3] = self.rbpf.rot_bar
        T_co[:3, 3] = self.rbpf.trans_bar

        self.sdf_optim.size_est = torch.tensor(self.size_est / self.ratio).to(
            self.device
        )

        T_co_opt, dist, _, size_est, loss_optim = self.sdf_optim.refine_pose(
            T_co, points_c, self.latent_tensor, steps=refine_steps, shape_only=False
        )
        # Let's do this multiple times
        for idx in range(2):
            self.latent_vector_prediction_pn(
                points_choice, T_co_opt[:3, :3], T_co_opt[:3, 3]
            )
            T_co_opt, dist, _, size_est, loss_optim = self.sdf_optim.refine_pose(
                T_co_opt,
                points_c,
                self.latent_tensor,
                steps=refine_steps,
                shape_only=False,
            )

        # debug information
        self.latent_vec_refine = self.latent_tensor.clone().detach()
        self.T_filter = T_co
        self.T_refine = T_co_opt

        self.rbpf.rot_bar = T_co_opt[:3, :3]
        self.rbpf.trans_bar = T_co_opt[:3, 3]

        return 0

    def initialize_latent_vector(self, points_c, T_co_init):
        dist_min = 100000000
        for _ in range(500):
            latent_size = 256
            std_ = 0.1
            rand_tensor = torch.ones(1, latent_size).normal_(mean=0, std=std_)
            latent_tensor_rand = rand_tensor.float().to(self.device)
            latent_tensor_rand.requires_grad = False
            dist = self.sdf_optim.eval_latent_vector(
                T_co_init, points_c, latent_tensor_rand
            )
            if dist < dist_min:
                self.latent_tensor = latent_tensor_rand.clone()
                dist_min = dist

    def sample_points(self, points_np, npoints=NPOINTS):
        if npoints < len(points_np):
            choice = (
                pointnet2_utils.furthest_point_sample(
                    torch.from_numpy(points_np)
                    .to(self.device)
                    .unsqueeze(0)
                    .contiguous(),
                    npoints,
                )
                .cpu()
                .numpy()[0]
            )
            points_choice = points_np[choice, :]
        else:
            choice = np.arange(0, len(points_np), dtype=np.int32)
            # print('length_point=', len(points_np))
            if npoints > len(points_np):
                lc = len(choice)
                le = npoints - len(points_np)
                if lc < le:
                    extra_choice = []
                    for i in range(le):
                        x = np.random.randint(lc)
                        extra_choice.append(x)
                    extra_choice = np.asarray(extra_choice)
                    choice = np.concatenate((choice, extra_choice), axis=0)
                else:
                    extra_choice = np.random.choice(choice, le, replace=False)
                    choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
            points_choice = points_np[choice, :]

        return points_choice
