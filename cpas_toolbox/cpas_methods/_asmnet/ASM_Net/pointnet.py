#
# This code is a modified version of the version available in the
# following repositories.
# https://github.com/isl-org/Open3D-PointNet
#

from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn


class STN3d(nn.Module):
    def __init__(self, num_points=2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(
                torch.from_numpy(
                    np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)
                )
            )
            .view(1, 9)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points=2500, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans


class ASM_Net(nn.Module):
    def __init__(self, num_points=2500, k=2):
        super(ASM_Net, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, global_feat=True)

        # asm branch
        self.ssm_fc1 = nn.Linear(1024, 512)
        self.ssm_fc2 = nn.Linear(512, 256)
        self.ssm_fc3 = nn.Linear(256, k)
        self.ssm_bn1 = nn.BatchNorm1d(512)
        self.ssm_bn2 = nn.BatchNorm1d(256)

        # pose branch
        self.pose_fc1 = nn.Linear(1024, 512)
        self.pose_fc2 = nn.Linear(512, 256)
        self.pose_fc3 = nn.Linear(256, 4)
        self.pose_bn1 = nn.BatchNorm1d(512)
        self.pose_bn2 = nn.BatchNorm1d(256)

        # activation func
        self.relu = nn.ReLU()

    def forward(self, x):
        # backbone
        x, trans = self.feat(x)

        # asm branch
        ssm_x = F.relu(self.ssm_bn1(self.ssm_fc1(x)))
        ssm_x = F.relu(self.ssm_bn2(self.ssm_fc2(ssm_x)))
        ssm_x = self.ssm_fc3(ssm_x)

        # pose branch
        pose_x = F.relu(self.pose_bn1(self.pose_fc1(x)))
        pose_x = F.relu(self.pose_bn2(self.pose_fc2(pose_x)))
        pose_x = self.pose_fc3(pose_x)
        pose_x = pose_x / torch.linalg.norm(pose_x, dim=1).unsqueeze(0).T
        return ssm_x, pose_x


if __name__ == "__main__":
    sim_data = Variable(torch.rand(32, 3, 2500))
    trans = STN3d()
    out = trans(sim_data)
    print("stn", out.size())

    pointfeat = PointNetfeat(global_feat=True)
    out, _ = pointfeat(sim_data)
    print("global feat", out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _ = pointfeat(sim_data)
    print("point feat", out.size())
