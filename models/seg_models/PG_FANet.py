# -*- coding: utf-8 -*-
# @Time    : 20/8/3 15:02
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : PG_FANet.py

""" Deep Feature Aggregation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.backbone import BACKBONE_DILATE
from module.blocks import ConvBNPReLU, Classifier_Module
import numpy as np
import random
from torch.distributions.uniform import Uniform


def _make_pred_layer(block, inplanes, dilation_series, padding_series, num_classes):
    return block(inplanes, dilation_series, padding_series, num_classes)


class BaseModel(nn.Module):
    def __init__(self, backbone, n_channels, num_classes):
        super(BaseModel, self).__init__()
        self.backbone = BACKBONE_DILATE[backbone](backbone=backbone, pretrained=True, dilate_scale=8)
        self.backbone_2 = BACKBONE_DILATE[backbone](backbone=backbone, pretrained=True, dilate_scale=8)
        del self.backbone_2.topconvs

        if int(backbone[6:]) > 34:
            dim = 2048
        else:
            dim = 512
        self.fusion_stage1_2 = nn.Conv2d(dim // 8 + 1, dim // 8, kernel_size=1, stride=1, bias=False)
        self.fusion_stage2_2 = nn.Conv2d(dim // 8 + 1, dim // 8, kernel_size=1, stride=1, bias=False)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pred_layer1 = _make_pred_layer(Classifier_Module, dim, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.pred_layer2 = _make_pred_layer(Classifier_Module, dim, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        self.enc2_1_reduce = ConvBNPReLU(dim // 8, 32, 1)
        self.enc2_2_reduce = ConvBNPReLU(dim // 8, 32, 1)

        self.enc2_3_reduce = ConvBNPReLU(dim // 4, 32, 1)
        self.enc2_4_reduce = ConvBNPReLU(dim // 4, 32, 1)

        self.enc2_5_reduce = ConvBNPReLU(dim // 2, 32, 1)
        self.enc2_6_reduce = ConvBNPReLU(dim // 2, 32, 1)

        self.enc2_7_reduce = ConvBNPReLU(dim, 32, 1)
        self.enc2_8_reduce = ConvBNPReLU(dim, 32, 1)

        self.conv_fusion = ConvBNPReLU(32, 32, 1)
        self.uni_dist = Uniform(-0.3, 0.3)
        self.fca_1_reduce = ConvBNPReLU(dim, 32, 1)
        self.fca_2_reduce = ConvBNPReLU(dim, 32, 1)
        self.random_dropout = nn.Dropout2d(p=0.5)
        self.conv_out = nn.Conv2d(32 + num_classes, num_classes, 1)
        self.conv_out1 = nn.Conv2d(num_classes, num_classes, 1)

    def get_backbone_layers(self):
        small_lr_layers = []
        return small_lr_layers

    def optim_parameters(self, lr):
        backbone_layer_id = [ii for m in self.get_backbone_layers() for ii in list(map(id, m.parameters()))]
        backbone_layer = filter(lambda p: id(p) in backbone_layer_id, self.parameters())
        rest_layer = filter(lambda p: id(p) not in backbone_layer_id, self.parameters())
        return [{'params': backbone_layer, 'lr': lr},
                {'params': rest_layer, 'lr': lr}]

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def dropout(self, x):
        return self.random_dropout(x)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, image):
        return


class PG_FANet(BaseModel):
    def __init__(self, backbone, n_channels, num_classes, **kwargs):
        super(PG_FANet, self).__init__(backbone, n_channels, num_classes)

    def get_backbone_layers(self):
        small_lr_layers = []
        small_lr_layers.append(self.backbone)
        small_lr_layers.append(self.backbone_2)
        return small_lr_layers

    def extract_feature(self, image, has_dropout=False):
        N, C, H, W = image.size()
        # ---------------------new dsn---------------------
        stage1_out_layers = self.backbone(image)
        x_stage1_att = self.pred_layer1(stage1_out_layers.layer4)
        x_prior = F.interpolate(x_stage1_att, size=stage1_out_layers.layer0.size()[2:], mode='bilinear',
                                align_corners=True)
        prob_maps = F.softmax(x_prior, dim=1)
        stage1_out = torch.argmax(prob_maps, dim=1).unsqueeze(dim=1)
        stage1_out[stage1_out >= 1] = 1.
        x_out_1 = F.interpolate(x_stage1_att, size=image.size()[2:], mode='bilinear',
                                align_corners=True)
        x_out_1 = self.conv_out1(x_out_1)
        # stage2
        stage1_to_stage2 = stage1_out_layers.layer0
        fusion_stage1_2 = self.fusion_stage1_2(torch.cat([stage1_to_stage2, stage1_out.float()], dim=1))
        fusion_stage1_2 = self.max_pool(fusion_stage1_2)
        stage2_enc1 = self.backbone_2.layer1(fusion_stage1_2)
        stage2_enc2 = self.backbone_2.layer2(stage2_enc1)
        stage2_enc3 = self.backbone_2.layer3(stage2_enc2)
        stage2_enc4 = self.backbone_2.layer4(stage2_enc3)
        x_stage2_att = self.pred_layer2(stage2_enc4)
        stage2_out = F.interpolate(x_stage2_att, size=stage1_out_layers.layer1.size()[2:], mode='bilinear',
                                   align_corners=True)
        # -----------multiscale fusion----------
        stage1_enc1_decoder = self.enc2_1_reduce(stage1_out_layers.layer1)
        stage2_enc1_docoder = self.enc2_2_reduce(stage2_enc1)

        stage1_enc2_decoder = self.enc2_3_reduce(stage1_out_layers.layer2)
        stage1_enc2_decoder = F.interpolate(stage1_enc2_decoder, size=stage1_out_layers.layer1.size()[2:],
                                            mode='bilinear',
                                            align_corners=True)
        stage2_enc2_docoder = self.enc2_4_reduce(stage2_enc2)
        stage2_enc2_docoder = F.interpolate(stage2_enc2_docoder, size=stage1_out_layers.layer1.size()[2:],
                                            mode='bilinear',
                                            align_corners=True)

        stage1_enc3_decoder = self.enc2_5_reduce(stage1_out_layers.layer3)
        stage1_enc3_decoder = F.interpolate(stage1_enc3_decoder, size=stage1_out_layers.layer1.size()[2:],
                                            mode='bilinear',
                                            align_corners=True)
        stage2_enc3_docoder = self.enc2_6_reduce(stage2_enc3)
        stage2_enc3_docoder = F.interpolate(stage2_enc3_docoder, size=stage1_out_layers.layer1.size()[2:],
                                            mode='bilinear',
                                            align_corners=True)

        fusion = stage1_enc1_decoder + stage2_enc1_docoder + \
                 stage1_enc2_decoder + stage2_enc2_docoder + \
                 stage1_enc3_decoder + stage2_enc3_docoder
        fusion = self.conv_fusion(fusion)
        # -----------multistage fusion----------
        stage1_fca_decoder = F.interpolate(self.fca_1_reduce(stage1_out_layers.layer4),
                                           size=stage1_out_layers.layer1.size()[2:],
                                           mode='bilinear', align_corners=True)
        stage2_fca_decoder = F.interpolate(self.fca_2_reduce(stage2_enc4), size=stage1_out_layers.layer1.size()[2:],
                                           mode='bilinear', align_corners=True)
        fusion = fusion + stage1_fca_decoder + stage2_fca_decoder
        cf = torch.cat([fusion, stage2_out], dim=1)
        if has_dropout:
            cf = self.dropout(cf)
        stage3_out_final = self.conv_out(cf)
        out = F.interpolate(stage3_out_final, size=image.size()[2:], mode='bilinear', align_corners=True)
        return out, (stage1_out_layers[-1], cf)

    def forward(self, image, has_dropout=False, random_noise=False):
        N, C, H, W = image.size()
        # ---------------------new dsn---------------------
        stage1_out_layers = self.backbone(image)
        x_stage1_att = self.pred_layer1(stage1_out_layers.layer4)
        x_prior = F.interpolate(x_stage1_att, size=stage1_out_layers.layer0.size()[2:], mode='bilinear',
                                align_corners=True)
        prob_maps = F.softmax(x_prior, dim=1)
        stage1_out = torch.argmax(prob_maps, dim=1).unsqueeze(dim=1)
        stage1_out[stage1_out >= 1] = 1.
        x_out_1 = F.interpolate(x_stage1_att, size=image.size()[2:], mode='bilinear',
                                align_corners=True)
        if random_noise:
            if random.random() > 0.5:
                x_out_1 = self.feature_dropout(x_out_1)
            if random.random() > 0.5:
                x_out_1 = self.feature_based_noise(x_out_1)
        x_out_1 = self.conv_out1(x_out_1)
        # stage2
        stage1_to_stage2 = stage1_out_layers.layer0
        fusion_stage1_2 = self.fusion_stage1_2(torch.cat([stage1_to_stage2, stage1_out.float()], dim=1))
        fusion_stage1_2 = self.max_pool(fusion_stage1_2)
        stage2_enc1 = self.backbone_2.layer1(fusion_stage1_2)
        stage2_enc2 = self.backbone_2.layer2(stage2_enc1)
        stage2_enc3 = self.backbone_2.layer3(stage2_enc2)
        stage2_enc4 = self.backbone_2.layer4(stage2_enc3)
        x_stage2_att = self.pred_layer2(stage2_enc4)
        stage2_out = F.interpolate(x_stage2_att, size=stage1_out_layers.layer1.size()[2:], mode='bilinear',
                                   align_corners=True)
        # -----------multiscale fusion----------
        stage1_enc1_decoder = self.enc2_1_reduce(stage1_out_layers.layer1)
        stage2_enc1_docoder = self.enc2_2_reduce(stage2_enc1)

        stage1_enc2_decoder = self.enc2_3_reduce(stage1_out_layers.layer2)
        stage1_enc2_decoder = F.interpolate(stage1_enc2_decoder, size=stage1_out_layers.layer1.size()[2:],
                                            mode='bilinear',
                                            align_corners=True)
        stage2_enc2_docoder = self.enc2_4_reduce(stage2_enc2)
        stage2_enc2_docoder = F.interpolate(stage2_enc2_docoder, size=stage1_out_layers.layer1.size()[2:],
                                            mode='bilinear',
                                            align_corners=True)

        stage1_enc3_decoder = self.enc2_5_reduce(stage1_out_layers.layer3)
        stage1_enc3_decoder = F.interpolate(stage1_enc3_decoder, size=stage1_out_layers.layer1.size()[2:],
                                            mode='bilinear',
                                            align_corners=True)
        stage2_enc3_docoder = self.enc2_6_reduce(stage2_enc3)
        stage2_enc3_docoder = F.interpolate(stage2_enc3_docoder, size=stage1_out_layers.layer1.size()[2:],
                                            mode='bilinear',
                                            align_corners=True)

        fusion = stage1_enc1_decoder + stage2_enc1_docoder + \
                 stage1_enc2_decoder + stage2_enc2_docoder + \
                 stage1_enc3_decoder + stage2_enc3_docoder
        fusion = self.conv_fusion(fusion)
        # -----------multistage fusion----------
        stage1_fca_decoder = F.interpolate(self.fca_1_reduce(stage1_out_layers.layer4),
                                           size=stage1_out_layers.layer1.size()[2:],
                                           mode='bilinear', align_corners=True)
        stage2_fca_decoder = F.interpolate(self.fca_2_reduce(stage2_enc4), size=stage1_out_layers.layer1.size()[2:],
                                           mode='bilinear', align_corners=True)
        fusion = fusion + stage1_fca_decoder + stage2_fca_decoder
        cf = torch.cat([fusion, stage2_out], dim=1)
        if has_dropout:
            cf = self.dropout(cf)
        stage3_out_final = self.conv_out(cf)
        out = F.interpolate(stage3_out_final, size=image.size()[2:], mode='bilinear', align_corners=True)
        return out, (stage1_out_layers[-1], x_out_1)


if __name__ == '__main__':
    with torch.no_grad():
        x = torch.rand((1, 3, 128, 128))
        model = PG_FANet_Twostage(backbone='resnet34',
                                  n_channels=3, num_classes=3)

