import glob
import os
import os.path
import random

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from PIL import Image
from transforms3d.euler import *
from transforms3d.quaternions import *

from ..pointnet2 import pointnet2_utils
from ..config.config import cfg
from ..utils.deepsdf_utils import *
from ..ycb_render.shapenet_renderer_tensor import *
from .data_augmenter import *


class shapenet_render_dataset(torch.utils.data.Dataset):
    def __init__(self, model_dir, model_ctg, gpu_id, render_size=128, output_size=(128, 128),
                 target_size=128, mode='train'):

        self.render_size = render_size
        self.renderer = ShapeNetTensorRenderer(self.render_size, self.render_size, gpu_id=gpu_id)
        self.h = render_size
        self.w = render_size
        self.output_size = output_size
        self.target_size = target_size

        self.model_ctg = model_ctg
        self.model_dir = model_dir

        self.model_paths = sorted(glob.glob(
                                    self.model_dir+'{}/{}/*.ply'.format(self.model_ctg, mode)))
        
        # show target
        if mode == 'val':
            self.model_paths = [self.model_dir+'{}/train/0000.ply'.format(self.model_ctg)] + self.model_paths

        self.models_num = len(self.model_paths)
        self.models_names = []
        for i in range(len(self.model_paths)):
            self.models_names.append(self.model_paths[i].split('/')[-1][:4])

        texture_paths = ['' for cls in self.models_names]

        self.renderer.load_objects(self.model_paths, texture_paths)
        
        # renderer properties
        self.renderer.set_projection_matrix(self.w, self.h, cfg.TRAIN.FU, cfg.TRAIN.FU,
                                            render_size/2.0, render_size/2.0, 0.01, 10)
        self.renderer.set_camera_default()
        self.renderer.set_light_pos([0, 0, 0])
        _, self.pose_list = torch.load('./data_files/pose2.pth')
        self.render_dist = cfg.TRAIN.RENDER_DIST[0]
        
        # put in the configuration file
        self.lb_shift = cfg.TRAIN.SHIFT_MIN
        self.ub_shift = cfg.TRAIN.SHIFT_MAX
        self.lb_scale = cfg.TRAIN.SCALE_MIN
        self.ub_scale = cfg.TRAIN.SCALE_MAX

        self.std_shift = cfg.TRAIN.SHIFT_STD
        self.std_scale = cfg.TRAIN.SCALE_STD

        self.noise_params = {
            # Multiplicative noise
            'gamma_shape': 1000.,
            'gamma_scale': 0.001,

            # Additive noise
            'gaussian_scale': 0.005,  # 5mm standard dev
            'gp_rescale_factor': 4,

            # Random ellipse dropout
            'ellipse_dropout_mean': 10,
            'ellipse_gamma_shape': 5.0,
            'ellipse_gamma_scale': 1.0,
        }

    def __getitem__(self, image_index):
        depth_input, depth_target, mask = self.load(image_index)

        add_noise_to_depth_cuda(depth_input, self.noise_params)

        shift = [np.random.uniform(self.lb_shift, self.ub_shift), np.random.uniform(self.lb_shift, self.ub_shift)]
        scale = np.random.uniform(self.lb_scale, self.ub_scale)

        # shift = [np.random.normal(0.0, self.std_shift), np.random.normal(0.0, self.std_shift)]
        # scale = np.random.normal(1.0, self.std_scale)

        shift = np.array(shift, dtype=np.float)
        affine = torch.from_numpy(np.float32([[scale, 0, shift[0] / self.h], [0, scale, shift[1] / self.w]])).float()
        shift = (shift - self.lb_shift)/(self.ub_shift - self.lb_shift)  # normalize to [0, 1]
        scale = (scale - self.lb_scale)/(self.ub_scale - self.lb_scale)  # normalize to [0, 1]

        # drop out
        mask_np = mask[:, :, 0].cpu().numpy()
        mask_np = dropout_random_ellipses_mask(mask_np, self.noise_params)
        mask_drop = torch.from_numpy(mask_np).cuda().unsqueeze(2).float()

        depth_input = depth_input * mask_drop
        
        return depth_input.permute(2, 0, 1), depth_target.permute(2, 0, 1), affine, shift, scale, mask.permute(2, 0, 1).float()

    # @profile
    def load(self, index):
        instance = random.sample(set(list(range(1, len(self.model_paths)))), 1)
        #print('instance id=', instance[0])
        self.renderer.instances = instance
        self.renderer.set_light_pos([0, 0, 0])
        self.renderer.set_light_color([1.5, 1.5, 1.5])

        # pose
        poses = [self.pose_list[index].cpu().numpy()]
        poses[0][0:3] = [0, 0, self.render_dist]
        d_euler = np.random.uniform(-10*np.pi/180, 10*np.pi/180, (3,))
        d_quat = euler2quat(d_euler[0], d_euler[1], d_euler[2])
        poses[0][3:] = qmult(poses[0][3:], d_quat)

        # declare cuda tensor
        frames_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
        seg_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
        frames_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
        seg_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
        pc_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
        pc_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)

        self.renderer.set_poses(poses)
        pose = self.renderer.get_poses()
        poses_cam = np.array(pose[0])

        self.renderer.render([0], frames_cuda, seg_cuda, pc2_tensor=pc_cuda)
        frames_cuda = frames_cuda.flip(0)
        seg_cuda = seg_cuda.flip(0)
        frames_cuda = frames_cuda[:, :, :3]  # get rid of normalization for adding noise
        seg = seg_cuda[:, :, :3]
        pc_cuda = pc_cuda.flip(0)
        pc_cuda = pc_cuda[:, :, :3]
        seg_input = seg[:, :, 0].clone()


        # for fixed condition:
        self.renderer.instances = [0]
        self.renderer.render([0], frames_target_cuda, seg_target_cuda, pc2_tensor=pc_target_cuda)
        frames_target_cuda = frames_target_cuda.flip(0)
        frames_target_cuda = frames_target_cuda[:, :, :3].float()
        seg_target_cuda = seg_target_cuda.flip(0)
        seg_target = seg_target_cuda[:, :, 0].clone().cpu().numpy()
        pc_target_cuda = pc_target_cuda.flip(0)
        pc_target_cuda = pc_target_cuda[:, :, :3]

        depth_input = pc_cuda[:, :, [2]]
        depth_target = pc_target_cuda[:, :, [2]]

        # normalize depth
        depth_input[depth_input > 0] = (depth_input[depth_input > 0] - self.render_dist) + 0.5
        depth_target[depth_target > 0] = (depth_target[depth_target > 0] - self.render_dist) + 0.5

        mask = (seg_input != 0).unsqueeze(2)

        return (depth_input, depth_target, mask)

    def __len__(self):
        return len(self.pose_list)

# render on the fly
class shapenet_codebook_online_generator(torch.utils.data.Dataset):
    def __init__(self, model_dir, model_ctg, gpu_id, render_size=128, output_size=(128, 128),
                 target_size=128):

        self.render_size = render_size
        self.renderer = ShapeNetTensorRenderer(self.render_size, self.render_size, gpu_id=gpu_id)
        self.h = render_size
        self.w = render_size
        self.output_size = output_size
        self.target_size = target_size

        self.model_ctg = model_ctg
        self.model_dir = model_dir


        self.model_path = self.model_dir + '{}/train/0000.ply'.format(self.model_ctg)
        texture_paths = ['']

        self.renderer.load_objects([self.model_path], texture_paths)

        # renderer properties
        self.renderer.set_projection_matrix(self.w, self.h, cfg.TRAIN.FU, cfg.TRAIN.FU,
                                            render_size / 2.0, render_size / 2.0, 0.01, 10)
        self.renderer.set_camera_default()
        self.renderer.set_light_pos([0, 0, 0])
        _, self.pose_list = torch.load('./data_files/codebook_pose.pth')
        self.render_dist = cfg.TRAIN.RENDER_DIST[0]

    def __getitem__(self, image_index):

        pose_cam, depth_target = self.load(image_index)
        depth_target = depth_target.permute(2, 0, 1)

        return pose_cam, depth_target

    # @profile
    def load(self, index):
        # randomize
        while True:
            self.renderer.set_light_pos([0, 0, 0])
            target_color = [1.0, 1.0, 1.0]
            self.renderer.set_light_color(target_color)

            # end randomize
            poses = [self.pose_list[index].cpu().numpy()]

            # declare cuda tensor
            frames_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
            seg_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
            pc_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
            poses[0][0:3] = [0, 0, self.render_dist]
            self.renderer.set_poses(poses)
            pose = self.renderer.get_poses()

            if np.max(np.abs(pose)) > 10:
                pose = [np.array([0,0,0,1,0,0,0]), np.array([0,0,0,1,0,0,0]), np.array([0,0,0,1,0,0,0])]
            poses_cam = np.array(pose)

            self.renderer.render([0], frames_target_cuda, seg_target_cuda, pc2_tensor=pc_target_cuda)
            frames_target_cuda = frames_target_cuda.flip(0)
            frames_target_cuda = frames_target_cuda[:, :, :3].float()
            seg_target_cuda = seg_target_cuda.flip(0)
            seg = seg_target_cuda[:, :, 0]

            pc_target_cuda = pc_target_cuda.flip(0)
            pc_target_cuda = pc_target_cuda[:, :, :3]
            depth_target = pc_target_cuda[:, :, [2]]

            depth_target[depth_target > 0] = (depth_target[depth_target > 0] - self.render_dist) + 0.5

            if torch.max(seg_target_cuda).data > 0:
                break

        return poses_cam.astype(np.float32), depth_target

    def __len__(self):
        return len(self.pose_list)

class pointnet_render_dataset(torch.utils.data.Dataset):
    def __init__(self, model_dir, model_ctg, gpu_id, render_size=128, output_size=(128, 128),
                 target_size=128, mode='train'):

        self.render_size = render_size
        self.renderer = ShapeNetTensorRenderer(self.render_size, self.render_size, gpu_id=gpu_id)
        self.h = render_size
        self.w = render_size
        self.output_size = output_size
        self.target_size = target_size

        self.model_ctg = model_ctg
        self.model_dir = model_dir

        self.model_paths = sorted(glob.glob(
                                    self.model_dir+'{}/{}/objects/*'.format(self.model_ctg, mode)))
        self.label_paths = sorted(glob.glob(
                                    self.model_dir+'{}/{}/labels/*.pth'.format(self.model_ctg, mode)))

        self.models_num = len(self.label_paths)
        self.models_names = []
        for i in range(len(self.model_paths)):
            self.models_names.append(self.model_paths[i].split('/')[-1])

        texture_paths = ['' for cls in self.models_names]
        
        self.renderer.load_objects_pn(self.model_paths, texture_paths)
        
        # renderer properties
        self.renderer.set_projection_matrix(self.w, self.h, cfg.TRAIN.FU, cfg.TRAIN.FV,
                                            render_size/2.0, render_size/2.0, 0.01, 10)
        self.renderer.set_camera_default()
        self.renderer.set_light_pos([0, 0, 0])
        _, self.pose_list = torch.load('./data_files/pose2.pth')
        self.render_dist = cfg.TRAIN.RENDER_DIST[0]

        # put in the configuration file
        self.lb_shift = cfg.TRAIN.SHIFT_MIN
        self.ub_shift = cfg.TRAIN.SHIFT_MAX
        self.lb_scale = cfg.TRAIN.SCALE_MIN
        self.ub_scale = cfg.TRAIN.SCALE_MAX

        self.std_shift = cfg.TRAIN.SHIFT_STD
        self.std_scale = cfg.TRAIN.SCALE_STD
        self.intrinsics = np.identity(3)
        self.intrinsics[0, 0] = cfg.TRAIN.FU
        self.intrinsics[1, 1] = cfg.TRAIN.FV
        self.intrinsics[0, 2] = cfg.TRAIN.U0
        self.intrinsics[1, 2] = cfg.TRAIN.V0
      
        self.noise_params = {
            # Multiplicative noise
            'gamma_shape': 1000.,
            'gamma_scale': 0.001,

            # Additive noise
            'gaussian_scale': 0.005,  # 5mm standard dev
            'gp_rescale_factor': 4,

            # Random ellipse dropout
            'ellipse_dropout_mean': 10,
            'ellipse_gamma_shape': 5.0,
            'ellipse_gamma_scale': 1.0,
        }

        self.rot_backprojection = np.eye(3)
        self.trans_backprojection = np.zeros(3)
        
    def __getitem__(self, image_index):
        depth_input, depth_target, mask, label = self.load(image_index)
        depth_clean = depth_input*1.0
        add_noise_to_depth_cuda(depth_input, self.noise_params)

        shift = [np.random.uniform(self.lb_shift, self.ub_shift), np.random.uniform(self.lb_shift, self.ub_shift)]
        scale = np.random.uniform(self.lb_scale, self.ub_scale)

        shift = np.array(shift, dtype=np.float)
        affine = torch.from_numpy(np.float32([[scale, 0, shift[0] / self.h], [0, scale, shift[1] / self.w]])).float()
        shift = (shift - self.lb_shift)/(self.ub_shift - self.lb_shift)  # normalize to [0, 1]
        scale = (scale - self.lb_scale)/(self.ub_scale - self.lb_scale)  # normalize to [0, 1]

        # drop out
        mask_np = mask[:, :, 0].cpu().numpy()
        mask_np = dropout_random_ellipses_mask(mask_np, self.noise_params)
        mask_drop = torch.from_numpy(mask_np).cuda().unsqueeze(2).float()

        depth_input = depth_input * mask_drop
        depth_np = depth_input.cpu().numpy()[:, :, 0]
        points_np = depth2pc(depth_np, depth_np.shape[0], depth_np.shape[1], self.intrinsics)[:,:3]
        
        npoints = 4873
        if len(points_np) > 0:
            points_choice = self.sample_points(points_np, npoints)
        else:
            depth_clean_np = depth_target.cpu().numpy()[:, :, 0]
            points_clean_np = depth2pc(depth_clean_np, depth_clean_np.shape[0], depth_clean_np.shape[1], self.intrinsics)[:,:3]
            points_choice = self.sample_points(points_clean_np, npoints)
        
        rot = self.rot_backprojection
        trans = self.trans_backprojection
        quat = mat2quat(rot)
        # add noise to orientation/rotation
        d_euler = np.random.uniform(-5.0*np.pi/180, 5.0*np.pi/180, (3,))
        d_quat = euler2quat(d_euler[0], d_euler[1], d_euler[2])
        quat_perturb = qmult(quat, d_quat)
        rot_perturb = quat2mat(quat_perturb)

        points_obj_before = np.dot(rot.T, (points_np.T - np.tile(trans, (len(points_np),1)).T)).T   # for visualize comparison 
        points_obj_before = np.float32(points_obj_before)

        points_obj = np.dot(rot_perturb.T, (points_choice.T - np.tile(trans, (len(points_choice),1)).T)).T
        points_obj = np.float32(points_obj)

        points_c = torch.from_numpy(points_obj).cuda()

        return depth_input.permute(2, 0, 1), depth_target.permute(2, 0, 1), label[0], points_c, affine, shift, scale, mask.permute(2, 0, 1).float()

    # @profile
    def load(self, index):
        instance = random.sample(set(list(range(0, len(self.model_paths)))), 1)
        label = torch.load(self.label_paths[instance[0]])[0,:,:].detach()

        self.renderer.instances = instance
        self.renderer.set_light_pos([0, 0, 0])
        self.renderer.set_light_color([1.5, 1.5, 1.5])

        # pose
        poses = [self.pose_list[index].cpu().numpy()]
        # poses = [self.pose_list[0].cpu().numpy()]
        poses[0][0:3] = [0, 0, self.render_dist]
        d_euler = np.random.uniform(-5.0*np.pi/180, 5.0*np.pi/180, (3,))
        d_quat = euler2quat(d_euler[0], d_euler[1], d_euler[2])
        poses[0][3:] = qmult(poses[0][3:], d_quat)

        # declare cuda tensor
        frames_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
        seg_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
        frames_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
        seg_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
        pc_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)
        pc_target_cuda = torch.cuda.FloatTensor(self.h, self.w, 4)

        self.renderer.set_poses(poses)
        pose = self.renderer.get_poses()
        poses_cam = np.array(pose[0])
        self.renderer.render([0], frames_cuda, seg_cuda, pc2_tensor=pc_cuda)
        frames_cuda = frames_cuda.flip(0)
        seg_cuda = seg_cuda.flip(0)
        frames_cuda = frames_cuda[:, :, :3]  # get rid of normalization for adding noise
        seg = seg_cuda[:, :, :3]
        pc_cuda = pc_cuda.flip(0)
        pc_cuda = pc_cuda[:, :, :3]
        seg_input = seg[:, :, 0].clone()

        self.rot_backprojection = self.renderer.poses_rot[0][:3, :3]
        self.trans_backprojection = self.renderer.poses_trans[0][-1, :3]
        
        # fixed condition for recover some of the bad validation data:
        self.renderer.instances = instance
        index_use = 122955  # new added recover pose
        poses_recover = [self.pose_list[index_use].cpu().numpy()]
        poses_recover[0][0:3] = [0, 0, self.render_dist]
        self.renderer.set_poses(poses_recover)
        self.renderer.render([0], frames_target_cuda, seg_target_cuda, pc2_tensor=pc_target_cuda)
        frames_target_cuda = frames_target_cuda.flip(0)
        frames_target_cuda = frames_target_cuda[:, :, :3].float()
        seg_target_cuda = seg_target_cuda.flip(0)
        seg_target = seg_target_cuda[:, :, 0].clone().cpu().numpy()
        pc_target_cuda = pc_target_cuda.flip(0)
        pc_target_cuda = pc_target_cuda[:, :, :3]

        depth_input = pc_cuda[:, :, [2]]
        depth_target = pc_target_cuda[:, :, [2]]

        mask = (seg_input != 0).unsqueeze(2)

        return (depth_input, depth_target, mask, label)

    def __len__(self):
        return len(self.pose_list)

    def sample_points(self, points_np, npoints):
        if npoints < len(points_np):
            choice = pointnet2_utils.furthest_point_sample(torch.from_numpy(points_np).cuda().unsqueeze(0).contiguous(), npoints).cpu().numpy()[0]
            points_choice = points_np[choice,:]
        else:
            choice = np.arange(0, len(points_np), dtype=np.int32)
            #print('length_point=', len(points_np))
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
                    extra_choice = np.random.choice(choice, le, replace = False)
                    choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
            points_choice = points_np[choice,:]  

        return points_choice
