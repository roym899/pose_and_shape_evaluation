import torch
import torch.nn as nn

from ..config import FLAGS
from .fs_net_repo.PoseNet9D import PoseNet9D
from .point_sample.face_sample import Sketch2Pc
from .pspnet import PSPNet

# from tools.shape_prior_utils import get_nocs_model, get_point_depth_error


# from tools.training_utils import get_gt_v


class SelfSketchPoseNet(nn.Module):
    def __init__(self, train_stage):
        super(SelfSketchPoseNet, self).__init__()
        self.posenet = PoseNet9D()
        self.seman_encoder = PSPNet(output_channel=FLAGS.feat_seman)
        self.train_stage = train_stage

    def forward(
        self,
        depth,
        obj_id,
        camK,
        mean_shape,
        rgb=None,
        shape_prior=None,
        def_mask=None,
        gt_2D=None,
        depth_normalize=None,
    ):
        output_dict = {}
        if FLAGS.use_seman_feat:
            if self.encoder_input_form == "rgb":
                seman_feature, obj_mask = self.seman_encoder(rgb)
            else:
                assert self.encoder_input_form == "rgbd"
                seman_feature, obj_mask = self.seman_encoder(rgb, depth_normalize)
            seman_feature = seman_feature.permute(0, 2, 3, 1)
            if torch.rand(1) < FLAGS.drop_seman_prob:
                seman_feature = torch.zeros_like(seman_feature).detach()
            obj_mask_output = torch.softmax(obj_mask, dim=1)
            # obj_mask_output = torch.exp(obj_mask)
        else:
            bs = depth.shape[0]
            H, W = depth.shape[2], depth.shape[3]
            seman_feature = torch.zeros((bs, H, W, 32)).float().to(rgb.device)
            obj_mask_output = obj_mask = None
        # RGB FEATURE FROM GNS / PSP
        bs = depth.shape[0]
        H, W = depth.shape[2], depth.shape[3]

        FLAGS.sample_method = "basic"

        sketch = torch.rand([bs, 6, H, W], device=depth.device)

        PC, PC_sk, PC_seman, PC_nocs = Sketch2Pc(
            sketch, def_mask, depth, camK, gt_2D, seman_feature
        )
        is_data_valid = True

        if PC is None or (not is_data_valid):
            return output_dict, None

        if PC.isnan().any():
            print("nan detect in point cloud!!")
            return output_dict, None

        PC = PC.detach()
        gt_s = None
        (
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
        ) = self.posenet(PC, obj_id, shape_prior, PC_seman, mean_shape, gt_s=gt_s)

        output_dict["recon_model"] = shape_prior + deform_field

        if Pred_T.isnan().any() or p_green_R.isnan().any() or p_red_R.isnan().any():
            print("nan detect in trans / rot!!")
            return output_dict, None

        if not FLAGS.use_point_conf_for_vote:
            raise NotImplementedError

        output_dict["mask"] = obj_mask_output
        output_dict["p_green_R"] = p_green_R
        output_dict["p_red_R"] = p_red_R
        output_dict["f_green_R"] = f_green_R
        output_dict["f_red_R"] = f_red_R
        output_dict["Pred_T"] = Pred_T
        output_dict["Pred_s"] = Pred_s
        return output_dict
