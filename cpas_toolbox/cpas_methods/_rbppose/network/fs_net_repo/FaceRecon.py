# follow FS-Net
import torch
import torch.nn as nn

from ...config import FLAGS

# global feature num : the channels of feature from rgb and depth
# grid_num : the volume resolution


class FaceRecon(nn.Module):
    def __init__(self):
        super(FaceRecon, self).__init__()
        self.neighbor_num = FLAGS.gcn_n_num
        self.support_num = FLAGS.gcn_sup_num
        # 3D convolution for point cloud
        self.recon_num = 3
        self.face_recon_num = FLAGS.face_recon_c

        dim_fuse = sum([128, 128, 256, 256, 512, FLAGS.obj_c, FLAGS.feat_seman])
        # 16: total 6 categories, 256 is global feature
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(dim_fuse, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.recon_head = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, self.recon_num, 1),
        )

        self.face_decoder = nn.Sequential(
            nn.Conv1d(FLAGS.feat_face + 3, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.vote_head_1 = VoteHead()
        self.vote_head_2 = VoteHead()
        self.vote_head_3 = VoteHead()
        self.vote_head_4 = VoteHead()
        self.vote_head_5 = VoteHead()
        self.vote_head_6 = VoteHead()
        self.vote_head_list = [
            self.vote_head_1,
            self.vote_head_2,
            self.vote_head_3,
            self.vote_head_4,
            self.vote_head_5,
            self.vote_head_6,
        ]
        self.mask_head = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        feat: "tensor (bs, vetice_num, 256)",
        feat_global: "tensor (bs, 1, 256)",
        vertices: "tensor (bs, vetice_num, 3)",
        face_shift_prior: "tensor (bs, vetice_num, 18)",
    ):
        """
        Return: (bs, vertice_num, class_num)
        """
        #  concate feature
        bs, vertice_num, _ = feat.size()
        feat_face_re = (
            feat_global.view(bs, 1, feat_global.shape[1])
            .repeat(1, feat.shape[1], 1)
            .permute(0, 2, 1)
        )
        conv1d_input = feat.permute(0, 2, 1)  # (bs, fuse_ch, vertice_num)
        conv1d_out = self.conv1d_block(conv1d_input)

        recon = self.recon_head(conv1d_out)
        # average pooling for face prediction
        feat_face_in = torch.cat(
            [feat_face_re, conv1d_out, vertices.permute(0, 2, 1)], dim=1
        )
        feat = self.face_decoder(feat_face_in)
        mask = self.mask_head(feat)
        face_shift_delta = torch.zeros((bs, vertice_num, 18)).to(feat.device)
        face_log_var = torch.zeros((bs, vertice_num, 6)).to(feat.device)
        for i, vote_head in enumerate(self.vote_head_list):
            face_vote_result = vote_head(
                feat, face_shift_prior[:, :, 3 * i : 3 * i + 3]
            )
            face_shift_delta[:, :, 3 * i : 3 * i + 3] = face_vote_result[:, :, :3]
            face_log_var[:, :, i] = face_vote_result[:, :, 3]

        return recon.permute(0, 2, 1), face_shift_delta, face_log_var, mask.squeeze()


class VoteHead(nn.Module):
    def __init__(self):
        super(VoteHead, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(256 + 3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 3 + 1, 1),
        )

    def forward(
        self,
        feat: "tensor (bs, 256, vertice_num)",
        face_shift_prior: "tensor (bs, vertice_num, 3)",
    ):
        """
        Return: (bs, vertice_num, class_num)
        """
        feat_face_in = torch.cat([feat, face_shift_prior.permute(0, 2, 1)], dim=1)
        face = self.layer(feat_face_in)
        return face.permute(0, 2, 1)
