import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import FLAGS
from ...tools.shape_prior_utils import (
    get_face_dis_from_nocs,
    get_face_shift_from_dis,
    get_nocs_from_deform,
)
from .FaceRecon import FaceRecon
from .pcl_encoder import PCL_Encoder
from .PoseR import RotHead
from .PoseTs import Pose_Ts, Pose_Ts_global


class PoseNet9D(nn.Module):
    def __init__(self, n_cat=6, prior_num=1024):
        super(PoseNet9D, self).__init__()
        self.n_cat = n_cat
        self.rot_green = RotHead()
        self.rot_red = RotHead()
        self.face_recon = FaceRecon()
        if FLAGS.use_global_feat_for_ts:
            self.ts = Pose_Ts_global()
        else:
            self.ts = Pose_Ts()
        self.pcl_encoder_obs = PCL_Encoder()
        # self.pcl_encoder_prior = self.pcl_encoder_obs
        self.assignment = nn.Sequential(
            nn.Conv1d(
                FLAGS.feat_pcl + 2 * FLAGS.feat_global_pcl + FLAGS.feat_seman, 512, 1
            ),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, self.n_cat * prior_num, 1),
        )
        self.deformation = nn.Sequential(
            nn.Conv1d(
                FLAGS.feat_pcl + 2 * FLAGS.feat_global_pcl + FLAGS.feat_seman, 512, 1
            ),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, self.n_cat * 3, 1),
        )

    def forward(self, points, obj_id, prior, PC_seman, mean_shape, gt_s=None):
        bs, p_num = points.shape[0], points.shape[1]
        prior_num = prior.shape[1]
        seman_dim = PC_seman.shape[2]
        feat_obs, feat_global_obs = self.pcl_encoder_obs(
            points - points.mean(dim=1, keepdim=True), obj_id
        )  # bs x p_num x 1286
        feat_prior, feat_global_prior = self.pcl_encoder_obs(prior, obj_id)

        feat_obs_fuse = torch.cat(
            (feat_obs, feat_global_obs.repeat(p_num, 1, 1).permute(1, 0, 2), PC_seman),
            dim=2,
        )  # bs x p_num x 1286
        # prior feature??
        feat_prior_fuse = torch.cat(
            (
                feat_prior,
                feat_global_prior.repeat(prior_num, 1, 1).permute(1, 0, 2),
                torch.zeros((bs, prior_num, seman_dim), device=prior.device),
            ),
            dim=2,
        )  # bs x p_num x 1286

        # translation and size
        if FLAGS.use_global_feat_for_ts:
            feat_for_ts = feat_global_obs
            T, s = self.ts(feat_for_ts)
        else:
            feat_for_ts = torch.cat(
                [feat_obs_fuse, points - points.mean(dim=1, keepdim=True)], dim=2
            )
            T, s = self.ts(feat_for_ts.permute(0, 2, 1))
        Pred_T = T + points.mean(dim=1)  # bs x 3
        Pred_s_delta = s  # this s is not the object size, it is the residual
        Pred_s = s + mean_shape

        # rotation, green for y axis, red for x
        green_R_vec = self.rot_green(feat_obs_fuse.permute(0, 2, 1))  # b x 4
        red_R_vec = self.rot_red(feat_obs_fuse.permute(0, 2, 1))  # b x 4
        # normalization
        p_green_R = green_R_vec[:, 1:] / (
            torch.norm(green_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6
        )
        p_red_R = red_R_vec[:, 1:] / (
            torch.norm(red_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6
        )
        # sigmoid for confidence
        f_green_R = F.sigmoid(green_R_vec[:, 0])
        f_red_R = F.sigmoid(red_R_vec[:, 0])

        if FLAGS.train:

            # assign matrix
            assign_feat = torch.cat(
                (feat_obs_fuse, feat_global_prior.repeat(p_num, 1, 1).permute(1, 0, 2)),
                dim=2,
            ).permute(
                0, 2, 1
            )  # bs x 2342 x n_pts
            assign_mat = self.assignment(assign_feat)
            assign_mat = assign_mat.view(
                -1, prior_num, p_num
            ).contiguous()  # bs, nc*nv, n_pts -> bs*nc, nv, n_pts
            index = (
                obj_id.long() + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
            )
            assign_mat = torch.index_select(assign_mat, 0, index)  # bs x nv x n_pts
            assign_mat = assign_mat.permute(0, 2, 1).contiguous()  # bs x n_pts x nv

            # deformation field
            deform_feat = torch.cat(
                (
                    feat_prior_fuse,
                    feat_global_obs.repeat(prior_num, 1, 1).permute(1, 0, 2),
                ),
                dim=2,
            ).permute(
                0, 2, 1
            )  # bs x 2342 x n_pts
            deform_field = self.deformation(deform_feat)
            deform_field = deform_field.view(
                -1, 3, prior_num
            ).contiguous()  # bs, nc*3, nv -> bs*nc, 3, nv
            deform_field = torch.index_select(deform_field, 0, index)  # bs x 3 x nv
            deform_field = deform_field.permute(0, 2, 1).contiguous()  # bs x nv x 3
            nocs_pred = get_nocs_from_deform(prior, deform_field, assign_mat)

            if FLAGS.prior_nocs_size == "gt":
                face_dis_prior = get_face_dis_from_nocs(nocs_pred, gt_s)
            elif FLAGS.prior_nocs_size == "pred":
                face_dis_prior = get_face_dis_from_nocs(nocs_pred, Pred_s)
            elif FLAGS.prior_nocs_size == "mean":
                face_dis_prior = get_face_dis_from_nocs(nocs_pred, mean_shape)
            else:
                raise NotImplementedError

            face_shift_prior = get_face_shift_from_dis(
                face_dis_prior,
                p_green_R,
                p_red_R,
                f_green_R,
                f_red_R,
                use_rectify_normal=FLAGS.use_rectify_normal,
            )
            if FLAGS.detach_prior_shift:
                face_shift_prior = face_shift_prior.detach()
            else:
                face_shift_prior = face_shift_prior

            recon, face_shift_delta, face_log_var, mask = self.face_recon(
                torch.cat((feat_obs, PC_seman), dim=2),
                feat_global_obs,
                points - points.mean(dim=1, keepdim=True),
                face_shift_prior,
            )
            recon = recon + points.mean(dim=1, keepdim=True)
            # handle face
            face_shift = face_shift_delta + face_shift_prior

            if not FLAGS.predict_uncertainty:
                face_log_var = torch.zeros_like(face_log_var)
            point_mask_conf = mask
        else:
            (
                recon,
                face_normal,
                face_shift,
                face_log_var,
                assign_mat,
                deform_field,
                nocs_pred,
                face_dis_delta,
                point_mask_conf,
                face_dis_prior,
                face_shift_delta,
            ) = (None, None, None, None, None, None, None, None, None, None, None)
            face_shift_prior = None

        if FLAGS.eval_coord:
            # assign matrix
            assign_feat = torch.cat(
                (feat_obs_fuse, feat_global_prior.repeat(p_num, 1, 1).permute(1, 0, 2)),
                dim=2,
            ).permute(
                0, 2, 1
            )  # bs x 2342 x n_pts
            assign_mat = self.assignment(assign_feat)
            assign_mat = assign_mat.view(
                -1, prior_num, p_num
            ).contiguous()  # bs, nc*nv, n_pts -> bs*nc, nv, n_pts
            index = (
                obj_id.long() + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
            )
            assign_mat = torch.index_select(assign_mat, 0, index)  # bs x nv x n_pts
            assign_mat = assign_mat.permute(0, 2, 1).contiguous()  # bs x n_pts x nv

            # deformation field
            deform_feat = torch.cat(
                (
                    feat_prior_fuse,
                    feat_global_obs.repeat(prior_num, 1, 1).permute(1, 0, 2),
                ),
                dim=2,
            ).permute(
                0, 2, 1
            )  # bs x 2342 x n_pts
            deform_field = self.deformation(deform_feat)
            deform_field = deform_field.view(
                -1, 3, prior_num
            ).contiguous()  # bs, nc*3, nv -> bs*nc, 3, nv
            deform_field = torch.index_select(deform_field, 0, index)  # bs x 3 x nv
            deform_field = deform_field.permute(0, 2, 1).contiguous()  # bs x nv x 3
            nocs_pred = get_nocs_from_deform(prior, deform_field, assign_mat)

        if FLAGS.eval_recon:
            # assign matrix
            # assign_feat = torch.cat((feat_obs_fuse, feat_global_prior.repeat(p_num, 1, 1).permute(1, 0, 2)),
            #                         dim=2).permute(0, 2, 1)  # bs x 2342 x n_pts
            # assign_mat = self.assignment(assign_feat)
            # assign_mat = assign_mat.view(-1, prior_num, p_num).contiguous()  # bs, nc*nv, n_pts -> bs*nc, nv, n_pts
            index = (
                obj_id.long() + torch.arange(bs, dtype=torch.long).cuda() * self.n_cat
            )
            # assign_mat = torch.index_select(assign_mat, 0, index)  # bs x nv x n_pts
            # assign_mat = assign_mat.permute(0, 2, 1).contiguous()  # bs x n_pts x nv

            # deformation field
            deform_feat = torch.cat(
                (
                    feat_prior_fuse,
                    feat_global_obs.repeat(prior_num, 1, 1).permute(1, 0, 2),
                ),
                dim=2,
            ).permute(
                0, 2, 1
            )  # bs x 2342 x n_pts
            deform_field = self.deformation(deform_feat)
            deform_field = deform_field.view(
                -1, 3, prior_num
            ).contiguous()  # bs, nc*3, nv -> bs*nc, 3, nv
            deform_field = torch.index_select(deform_field, 0, index)  # bs x 3 x nv
            deform_field = deform_field.permute(0, 2, 1).contiguous()  # bs x nv x 3
            # nocs_pred = get_nocs_from_deform(prior, deform_field, assign_mat)

        return (
            recon,
            face_shift,
            face_shift_delta,
            face_shift_prior,
            face_log_var,
            p_green_R,
            p_red_R,
            f_green_R,
            f_red_R,
            Pred_T,
            Pred_s,
            Pred_s_delta,
            assign_mat,
            deform_field,
            nocs_pred,
            point_mask_conf,
        )
