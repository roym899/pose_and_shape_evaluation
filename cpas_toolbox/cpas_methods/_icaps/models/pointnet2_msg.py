from collections import OrderedDict

import torch
import torch.nn as nn

from ..pointnet2 import pytorch_utils as pt_utils
from ..pointnet2.pointnet2_modules import PointnetSAModuleMSG


def get_model(input_channels=0):
    return Pointnet2MSG(input_channels=input_channels)


NPOINTS = [4096, 1024, 256, 64]
RADIUS = [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]
NSAMPLE = [[16, 32], [16, 32], [16, 32], [16, 32]]
MLPS = [
    [[16, 16, 32], [32, 32, 64]],
    [[64, 64, 128], [64, 96, 128]],
    [[128, 196, 256], [128, 196, 256]],
    [[256, 256, 512], [256, 384, 512]],
]
FP_MLPS = [[128, 128], [256, 256], [512, 512], [512, 512]]
CLS_FC = [512]  # [128]
DP_RATIO = 0.5


class Pointnet2MSG(nn.Module):
    def __init__(self, input_channels=6):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=NPOINTS[k],
                    radii=RADIUS[k],
                    nsamples=NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=True,
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        # self.FP_modules = nn.ModuleList()

        # for k in range(FP_MLPS.__len__()):
        #     pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
        #     self.FP_modules.append(
        #         PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k])
        #     )

        cls_layers = []
        # pre_channel = FP_MLPS[0][-1]
        # for k in range(0, CLS_FC.__len__()):
        #     cls_layers.append(pt_utils.Conv1d(pre_channel, CLS_FC[k], bn=True))
        #     pre_channel = CLS_FC[k]
        # cls_layers.append(pt_utils.Conv1d(pre_channel, 256, activation=None))

        pre_channel = (MLPS[-1][0][-1] + MLPS[-1][1][-1]) * NPOINTS[-1]
        for k in range(0, CLS_FC.__len__()):
            cls_layers.append(pt_utils.FC(pre_channel, CLS_FC[k], bn=True))
            pre_channel = CLS_FC[k]
        cls_layers.append(pt_utils.FC(pre_channel, 256, activation=None))

        cls_layers.insert(1, nn.Dropout(DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        # for i in range(-1, -(len(self.FP_modules) + 1), -1):
        #     l_features[i - 1] = self.FP_modules[i](
        #         l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
        #     )

        # pred_cls = self.cls_layer(l_features[0]).transpose(1, 2).contiguous()  # (B, N, 1)

        fc_feature = torch.flatten(l_features[-1], 1)
        pred_cls = self.cls_layer(fc_feature)  # (B, N, 1)

        return pred_cls

    def load_ckpt_weights(self, state_dict):
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     m_idx = isSubstring('module', str(k))
        #     if m_idx == -1:
        #         name = k
        #     else:
        #         name = k[:m_idx] + k[m_idx+7:]

        #     new_state_dict[name] = v

        self.load_state_dict(state_dict)
        self.weights_loaded = True


class BootstrapedMSEloss(nn.Module):
    def __init__(self, b=200):
        super(BootstrapedMSEloss, self).__init__()
        self.b = b

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        batch_size = pred.size(0)
        diff = torch.sum((target - pred) ** 2, 1)
        diff = diff.view(batch_size, -1)
        # diff= torch.topk(diff, self.b, dim=1)
        self.loss = diff[:, 0].mean()
        return self.loss


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
