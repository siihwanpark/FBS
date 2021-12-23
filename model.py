import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import global_avgpool2d, winner_take_all


class FBSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, fbs=False, sparsity_ratio=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.fbs = fbs
        self.sparsity_ratio = sparsity_ratio

        if fbs:
            self.channel_saliency_predictor = nn.Linear(in_channels, out_channels)
            nn.init.kaiming_normal_(self.channel_saliency_predictor.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.channel_saliency_predictor.bias, 1.)

            self.bn.weight.requires_grad_(False)

    def forward(self, x, inference=False):
        if self.fbs:
            x, g, active_channels = self.fbs_forward(x, inference)
            return x, g, active_channels

        else:
            x = self.original_forward(x)
            return x

    def original_forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

    def fbs_forward(self, x, inference):
        ss = global_avgpool2d(x) # [batch, C1, H1, W1] -> [batch, C1]
        g = self.channel_saliency_predictor(ss) # [batch, C1] -> [batch, C2]
        pi = winner_take_all(g, self.sparsity_ratio) # [batch, C2]
        x = self.conv(x)  # [batch, C1, H1, W1] -> [batch, C2, H2, W2]

        if inference:
            ones, zeros = torch.ones_like(pi), torch.zeros_like(pi)
            pre_mask = torch.where(pi != 0, ones, zeros)
            pre_mask = pre_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.size(2), x.size(3))
            x = x * pre_mask

        active_channels = (x.abs().sum(dim=-1).sum(dim=-1) > 1e-15).float()
        x = self.bn(x)
        post_mask = pi.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.size(2), x.size(3))
        x = x * post_mask
        x = F.relu(x)
        
        return x, torch.mean(torch.sum(g, dim = -1)), active_channels # E_x[||g_l(x_l-1)||_1]


class CifarNet(nn.Module):
    def __init__(self, fbs=False, sparsity_ratio=1.0):
        super().__init__()
        self.layer0 = FBSConv2d(3, 64, 3, stride=1, padding=0, fbs=fbs, sparsity_ratio=sparsity_ratio)
        self.layer1 = FBSConv2d(64, 64, 3, stride=1, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio)
        self.layer2 = FBSConv2d(64, 128, 3, stride=2, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio)
        self.layer3 = FBSConv2d(128, 128, 3, stride=1, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio)
        self.layer4 = FBSConv2d(128, 128, 3, stride=1, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio)
        self.layer5 = FBSConv2d(128, 192, 3, stride=2, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio)
        self.layer6 = FBSConv2d(192, 192, 3, stride=1, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio)
        self.layer7 = FBSConv2d(192, 192, 3, stride=1, padding=1, fbs=fbs, sparsity_ratio=sparsity_ratio)

        self.pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(192, 10)

        self.fbs = fbs
        self.sparsity_ratio = sparsity_ratio

    # TODO: get g for each layer and calculate lasso
    def forward(self, x, inference = False):
        if self.fbs:
            active_channel_list = []

            lasso = 0.
            x, g, active_channels = self.layer0(x, inference)
            active_channel_list.append(active_channels)
            lasso += g

            x, g, active_channels = self.layer1(x, inference)
            active_channel_list.append(active_channels)
            lasso += g

            x, g, active_channels = self.layer2(x, inference)
            active_channel_list.append(active_channels)
            lasso += g

            x, g, active_channels = self.layer3(x, inference)
            active_channel_list.append(active_channels)
            lasso += g

            x, g, active_channels = self.layer4(x, inference)
            active_channel_list.append(active_channels)
            lasso += g

            x, g, active_channels = self.layer5(x, inference)
            active_channel_list.append(active_channels)
            lasso += g

            x, g, active_channels = self.layer6(x, inference)
            active_channel_list.append(active_channels)
            lasso += g

            x, g, active_channels = self.layer7(x, inference)
            active_channel_list.append(active_channels)
            lasso += g

            x = self.pool(x)
            x = torch.flatten(x, 1)

            x = self.fc(x)

            return x, lasso, active_channel_list

        else:
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.layer7(x)

            x = self.pool(x)
            x = torch.flatten(x, 1)
            
            x = self.fc(x)

            return x
