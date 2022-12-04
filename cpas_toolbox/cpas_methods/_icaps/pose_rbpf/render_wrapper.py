import time

import matplotlib.pyplot as plt
import numpy.ma as ma

from ..utils.poserbpf_utils import *
from ..ycb_render.shapenet_renderer_tensor import *


class render_wrapper:
    def __init__(self, model_ins, model_dir, gpu_id, scale=1.0, im_w=640, im_h=480):

        obj_paths = [model_dir + '{}.ply'.format(model_ins)]

        print(obj_paths)

        texture_paths = ['']

        self.cam_cx = 322.525
        self.cam_cy = 244.11084
        self.cam_fx = 591.0125
        self.cam_fy = 590.16775
        intrinsics = np.array([[self.cam_fx, 0, self.cam_cx], [0, self.cam_fy, self.cam_cy], [0, 0, 1]],
                                   dtype=np.float32)

        self.obj_paths = obj_paths
        self.texture_paths = texture_paths

        self.renderer = ShapeNetTensorRenderer(im_w, im_h, gpu_id=gpu_id)
        self.renderer.load_objects(obj_paths, texture_paths, scale=scale)
        self.renderer.set_light_pos(cfg.TRAIN.TARGET_LIGHT1_POS)
        self.renderer.set_light_color([1.0, 1.0, 1.0])
        self.renderer_cam_pos = [0, 0, 0]
        self.renderer.set_camera_default()
        self.renderer.set_projection_matrix(im_w, im_h,
                                            intrinsics[0, 0], intrinsics[1, 1],
                                            intrinsics[0, 2], intrinsics[1, 2], 0.01, 10)

        self.intrinsics = intrinsics
        self.im_w = im_w
        self.im_h = im_h

        # predefined parameters
        self.ymap_full = np.array([[j for i in range(int(self.im_w))] for j in range(self.im_h)])
        self.xmap_full = np.array([[i for i in range(int(self.im_w))] for j in range(self.im_h)])

        self.ymap_full_torch = torch.from_numpy(self.ymap_full).float().unsqueeze(2).cuda()
        self.xmap_full_torch = torch.from_numpy(self.xmap_full).float().unsqueeze(2).cuda()

    def set_intrinsics(self, intrinsics, im_w=640, im_h=480):
        self.renderer.set_camera_default()
        self.renderer.set_projection_matrix(im_w, im_h,
                                            intrinsics[0, 0], intrinsics[1, 1],
                                            intrinsics[0, 2], intrinsics[1, 2], 0.01, 10)
        self.intrinsics = intrinsics
        self.im_w = im_w
        self.im_h = im_h

    def render_pose(self, t, R): # cls_id: target object index in the loaded classes
        t_v = t
        q_v = mat2quat(R)
        pose_v = np.zeros((7,))
        pose_v[:3] = t_v
        pose_v[3:] = q_v
        frame_cuda = torch.cuda.FloatTensor(int(self.im_h), int(self.im_w), 4)
        seg_cuda = torch.cuda.FloatTensor(int(self.im_h), int(self.im_w), 4)
        pc_cuda = torch.cuda.FloatTensor(int(self.im_h), int(self.im_w), 4)
        self.renderer.set_poses([pose_v])
        self.renderer.render([0], frame_cuda, seg_cuda, pc2_tensor=pc_cuda)
        frame_cuda = frame_cuda.flip(0)
        frame_cuda = frame_cuda[:, :, :3].float()
        frame_cuda = frame_cuda.permute(2, 0, 1).unsqueeze(0)
        pc_cuda = pc_cuda.flip(0)
        depth_render = pc_cuda[:, :, 2]
        return frame_cuda, depth_render