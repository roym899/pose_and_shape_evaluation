import torch.utils.data as data
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import _pickle as cPickle

class nocs_real_dataset(data.Dataset):
    def __init__(self, obj_ctg, list_file, path, path_gt, real_model_dir, detection_prior_dir='../NOCS_dataset/real_test_cass_seg/'):
        self.path = path

        list_file = open(list_file)
        name_list = list_file.read().splitlines()

        self.obj_instance = name_list[0]
        file_start = name_list[1]
        file_end = name_list[2]

        start_num = int(file_start.split('/')[1])
        end_num = int(file_end.split('/')[1])

        self.scene_id = int(file_start.split('/')[0])

        file_num = np.arange(start_num, end_num+1)
        
        file_list = []
        file_gt_list = []
        self.file_names = []
        for i in range(len(file_num)):
            if not os.path.exists(path+'scene_' + file_start.split('/')[0]+'/{:04}_meta.txt'.format(file_num[i])):
                continue
            file_list.append('scene_' + file_start.split('/')[0]+'/{:04}'.format(file_num[i]))
            file_gt_list.append('results_real_test_scene_' + file_start.split('/')[0]+'_{:04}.pkl'.format(file_num[i]))
            self.file_names.append(file_start.split('/')[0]+'/{:04}/{}'.format(file_num[i], self.obj_instance))

        self.file_list = file_list

        self.files_rgb = [path + item + '_color.png' for item in file_list]
        self.files_depth = [path + item + '_depth.png' for item in file_list]
        self.files_mask = [detection_prior_dir + item + '_nocs_segmentation.png' for item in file_list] # self.files_mask = [path + item + '_mask.png' for item in file_list] #      
        self.files_meta = [path + item + '_meta.txt' for item in file_list]
        self.files_gt = [path_gt + item for item in file_gt_list]

        self.cam_cx = 322.525
        self.cam_cy = 244.11084
        self.cam_fx = 591.0125
        self.cam_fy = 590.16775
        self.intrinsics = np.array([[self.cam_fx, 0, self.cam_cx], [0, self.cam_fy, self.cam_cy], [0, 0, 1]],
                                   dtype=np.float32)

        # get GT scale:
        dim_info = real_model_dir + '{}.txt'.format(self.obj_instance)
        dim_list_file = open(dim_info)
        dim_list = dim_list_file.read().splitlines()
        dim_list_float = []
        for dim in dim_list:
            dim_list_float.append(float(dim))
        self.size = max(dim_list_float)

        # detection prior from CASS:
        self.files_bbox = [detection_prior_dir + item + '_nocs_bbox.txt' for item in file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # rgb image
        img = np.array(Image.open(self.files_rgb[idx]))

        # read semantic labels
        mask = np.array(Image.open(self.files_mask[idx]))
        mask = np.expand_dims(mask, 2)

        # read the depth image
        depth = np.array(self.load_depth(self.files_depth[idx]))/1000
        depth = np.expand_dims(depth, 2)

        # load pose
        pose, obj_idx = self.get_pose(idx, self.obj_instance)
        
        # import pdb;pdb.set_trace()
        fn = self.file_names[idx]

        # detection prior
        obj_all, bbox_all = self.load_bbox(self.files_bbox[idx])
        bbox_idx = [i for i, x in enumerate(obj_all) if x == obj_idx]
        bbox_obj = np.array([0, 0, 0, 0], dtype=np.float32)
        prior_uv = [0, 0]
        mask_obj = mask*1.0
        if len(bbox_idx) != 0:
            bbox_obj = bbox_all[bbox_idx[0]]
            prior_uv[0] = (bbox_obj[2] + bbox_obj[3])/2.0
            prior_uv[1] = (bbox_obj[0] + bbox_obj[1])/2.0
        if prior_uv[0] > 0 and prior_uv[1] > 0:
            patch = mask[int(prior_uv[1])-10:int(prior_uv[1])+10, int(prior_uv[0])-10:int(prior_uv[0])+10, 0].flatten()
            label = max(set(list(patch)), key=list(patch).count)
            mask_obj = (mask == int(label)).astype(np.float32)
            if label == 255:
                bbox_obj = np.array([0., 0., 0., 0.])

        return img, pose, self.intrinsics, depth, mask_obj, fn, self.size, bbox_obj

    def load_depth(self, depth_path):
        depth = cv2.imread(depth_path, -1)

        if len(depth.shape) == 3:
            depth16 = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2])
            depth16 = depth16.astype(np.uint16)
        elif len(depth.shape) == 2 and depth.dtype == 'uint16':
            depth16 = depth
        else:
            assert False, '[ Error ]: Unsupported depth type.'

        return depth16

    def load_bbox(self, filename):
        obj_ids = []
        bbox = []

        lines = [line.rstrip('\n') for line in open(filename)]
        for line in lines:
            line_split = line.split(' ')
            obj_ids.append(line_split[0])
            bbox.append([line_split[1], line_split[2], line_split[3], line_split[4]])

        return np.array(obj_ids, dtype=np.int), np.array(bbox, dtype=np.float32)

    def get_pose(self, idx, choose_obj):
        with open(self.files_gt[idx], 'rb') as f:
            nocs_data = cPickle.load(f)

        input_file = open(self.files_meta[idx], 'r')
        obj_idx = None
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            input_line = input_line.split(' ')
            if input_line[-1] == choose_obj:
                obj_idx = int(input_line[0])
                break
        input_file.close()
        assert obj_idx > 0, 'object is not in the image ! '

        if obj_idx - 1 < len(nocs_data['gt_RTs']):
            pose = nocs_data['gt_RTs'][obj_idx-1]
            pose[:3, :3] = pose[:3, :3] / np.cbrt(np.linalg.det(pose[:3, :3]))
        else:
            pose = np.eye(4)
        return pose, obj_idx