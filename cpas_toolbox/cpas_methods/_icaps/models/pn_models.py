import os
from collections import OrderedDict
from decimal import *

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
from config.config import cfg
from transforms3d.quaternions import *


def printProgressBar(
    iteration, total, prefix="", suffix="", decimals=1, length=100, fill="*"
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s \n" % (prefix, bar, percent, suffix))
    # # Print New Line on Complete
    # if iteration == total:
    #     print()


def isSubstring(s1, s2):
    M = len(s1)
    N = len(s2)

    # A loop to slide pat[] one by one
    for i in range(N - M + 1):

        # For current index i,
        # check for pattern match
        for j in range(M):
            if s2[i + j] != s1[j]:
                break

        if j + 1 == M:
            return i

    return -1


class encoder_depth(nn.Module):
    def __init__(self, capacity=1, code_dim=128):
        super(encoder_depth, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128 * capacity, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128 * capacity, 256 * capacity, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256 * capacity, 256 * capacity, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256 * capacity, 512 * capacity, 5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.fc = nn.Linear(512 * 8 * 8 * capacity, code_dim)

        self.capacity = capacity

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1, 512 * 8 * 8 * self.capacity)
        out = self.fc(out)
        return out


class decoder_depth(nn.Module):
    def __init__(self, capacity=1, code_dim=128):
        super(decoder_depth, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(
                256 * capacity, 128 * capacity, 5, 2, padding=2, output_padding=1
            ),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                128 * capacity, 128 * capacity, 5, 2, padding=2, output_padding=1
            ),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(
                128 * capacity, 64 * capacity, 5, 2, padding=2, output_padding=1
            ),
            nn.ReLU(),
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(64 * capacity, 1, 5, 2, padding=2, output_padding=1)
        )

        self.fc = nn.Linear(code_dim, 256 * 8 * 8 * capacity)

        self.dropout = nn.Dropout(0.5)

        self.capacity = capacity

    def forward(self, x):
        out = self.fc(x)
        out = out.view(-1, 256 * self.capacity, 8, 8)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.xavier_uniform_(m.weight)
        init.constant_(m.bias, 0.0)
    elif classname.find("Linear") != -1:
        init.xavier_uniform_(m.weight)
        init.constant_(m.bias, 0.0)


def pairwise_cosine_distances(x, y, eps=1e-8):
    """
    :param x: batch of code from the encoder (batch size x code size)
    :param y: code book (codebook size x code size)
    :return: cosine similarity matrix (batch size x code book size)
    """
    dot_product = torch.mm(x, torch.t(y))
    x_norm = torch.norm(x, 2, 1).unsqueeze(1)
    y_norm = torch.norm(y, 2, 1).unsqueeze(1)
    normalizer = torch.mm(x_norm, torch.t(y_norm))

    return dot_product / normalizer.clamp(min=eps)


class BootstrapedMSEloss(nn.Module):
    def __init__(self, b=200):
        super(BootstrapedMSEloss, self).__init__()
        self.b = b

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        batch_size = pred.size(0)
        diff = torch.sum((target - pred) ** 2, 1)
        diff = diff.view(batch_size, -1)
        diff = torch.topk(diff, self.b, dim=1)
        self.loss = diff[0].mean()
        return self.loss


class AAE(nn.Module):
    def __init__(
        self, object_names, capacity=1, code_dim=128, model_path=None, device="cuda"
    ):
        super(AAE, self).__init__()

        self.device = device
        self.object_names = object_names
        self.code_dim = code_dim

        self.encoder = encoder_depth(capacity=capacity, code_dim=code_dim)
        self.decoder = decoder_depth(capacity=capacity, code_dim=code_dim)

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)

        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=0.0002
        )

        self.model_path = model_path

        self.B_loss = BootstrapedMSEloss(cfg.TRAIN.BOOTSTRAP_CONST)
        self.L1_loss = torch.nn.L1Loss()
        self.Cos_loss = nn.CosineEmbeddingLoss()

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.B_loss = self.B_loss.to(self.device)
        self.Cos_loss = self.Cos_loss.to(self.device)

        # CPU Rendering
        self.CPU_Render = False

        self.angle_diff = np.array([])

    def load_ckpt_weights(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            m_idx = isSubstring("module", str(k))
            if m_idx == -1:
                name = k
            else:
                name = k[:m_idx] + k[m_idx + 7 :]

            new_state_dict[name] = v

        self.load_state_dict(new_state_dict)
        self.weights_loaded = True

    # codebook generation and saving
    def compute_codebook(self, code_dataset, save_dir, code_dim=128, save=True):
        assert self.weights_loaded, "need to load pretrained weights!"
        codebook_batch_size = 1000
        code_generator = torch.utils.data.DataLoader(
            code_dataset, batch_size=codebook_batch_size, shuffle=False, num_workers=0
        )
        print("code book size {}".format(len(code_dataset)))
        step = 0
        self.encoder.eval()

        codebook_cpt = torch.zeros(len(code_dataset), code_dim).to(self.device)
        codepose_cpt = torch.zeros(len(code_dataset), 7).to(self.device)
        for inputs in code_generator:
            poses, depth = inputs

            poses = poses.to(self.device)
            depth = depth.to(self.device)

            code = self.encoder.forward(depth).detach().view(depth.size(0), -1)

            print(code.size())

            codebook_cpt[
                step * codebook_batch_size : step * codebook_batch_size + code.size(0),
                :,
            ] = code
            codepose_cpt[
                step * codebook_batch_size : step * codebook_batch_size + code.size(0),
                :,
            ] = poses.squeeze(1)

            step += 1
            print("finished {}/{}".format(step, len(code_generator)))

        if save:
            torch.save((codebook_cpt, codepose_cpt), save_dir)
            print("code book is saved to {}".format(save_dir))

    # forward passing
    # normalized depth -> reconstruction + code
    def forward(self, x):
        code = self.encoder(x)
        out = self.decoder(code)
        return [out, code]

    def match_codebook(self, code, codebook, codepose):
        """
        compare the code with the codebook and retrieve the pose
        :param code: code from encoder (batch size x code size, e.g. 64x128)
        :return: pose retrieved from the codebook (batch size x 7(trans+rot), e.g. 64x7)
                code retrieved from codebook (batch size x code size, e.g. 64x128)
        """
        assert codebook.size(0) > 0 and codepose.size(0) == codebook.size(
            0
        ), "codebook is empty"
        distance_matrix = pairwise_cosine_distances(code, codebook)
        best_match = torch.argmax(distance_matrix, dim=1).cpu().numpy()
        code_recovered = codebook[best_match, :].to(self.device)
        pose_recovered = codepose[best_match, :]
        return pose_recovered, code_recovered

    def match_codebook_trans(self, code, codebook, codepose):
        """
        compare the code with the codebook and retrieve the pose
        :param code: code from encoder (batch size x code size, e.g. 64x128)
        :return: pose retrieved from the codebook (batch size x 7(trans+rot), e.g. 64x7)
                code retrieved from codebook (batch size x code size, e.g. 64x128)
        """
        assert codebook.size(0) > 0, "codebook is empty"
        distance_matrix = pairwise_cosine_distances(code, codebook)
        distance_max, idx = torch.max(distance_matrix, dim=1)
        distance_max = distance_max.cpu().numpy()
        rot = codepose[idx].cpu().numpy()
        return distance_max, rot

    def compute_distance_matrix(self, code, codebook):
        assert codebook.size(0) > 0, "codebook is empty"
        return pairwise_cosine_distances(code, codebook)

    def pairwise_distances(self, x, y=None):
        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

        return torch.clamp(dist, 0.0, np.inf)

    def retrieve_idx_q(self, q, codepose):
        q_query = codepose[:, 3:]
        distance_matrix = self.pairwise_distances(q, q_query)
        distance_max, idx = torch.min(distance_matrix, dim=1)
        return idx

    def retrieve_code_q(self, q, codebook, codepose):
        q_query = codepose[:, 3:]
        distance_matrix = self.pairwise_distances(q, q_query)
        distance_max, idx = torch.min(distance_matrix, dim=1)
        code_retrival = codebook[idx]
        pose_retrival = codepose[idx]

        return code_retrival, pose_retrival

    # todo: check if this doesn't crash for batch size > 1
    def match_codebook_topk(self, code, codebook, codepose, k=5):
        """
        compare the code with the codebook and retrieve the topk pose
        :param code: code from encoder (batch size x code size, e.g. 64x128)
        :return: pose retrieved from the codebook (batch size x 7(trans+rot), e.g. 64x7)
                code retrieved from codebook (batch size x code size, e.g. 64x128)
        """
        assert codebook.size(0) > 0 and codepose.size(0) == codebook.size(
            0
        ), "codebook is empty"
        distance_matrix = pairwise_cosine_distances(code, codebook)
        code_distance, best_match = torch.topk(distance_matrix, k, dim=1)
        code_distance = code_distance.detach().cpu().numpy()
        best_match = best_match.detach().cpu().numpy()
        code_recovered = codebook[best_match, :].to(self.device)
        pose_recovered = codepose[best_match, :]
        return pose_recovered, code_recovered, code_distance[0]

    def compute_pose_error(self, pose_est, pose_gt):
        update_image = False
        error_batch = np.array([])
        for q in range(pose_gt.size(0)):
            q_est = pose_est[q, 3:].cpu().numpy()
            q_gt = pose_gt[q, 3:].cpu().numpy()
            q_diff = qmult(qinverse(q_est), q_gt)
            _, d_angle = quat2axangle(q_diff)
            if d_angle > np.pi:
                d_angle = d_angle - 2 * np.pi
                d_angle = -d_angle
            self.angle_diff = np.append(self.angle_diff, d_angle / np.pi * 180)
            error_batch = np.append(error_batch, d_angle / np.pi * 180)
            if self.angle_diff[-1] > self.max_angle_error:
                self.max_angle_error = self.angle_diff[-1]
                self.max_angle_error_idx = q
                update_image = True
        return update_image, error_batch
