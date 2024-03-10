# -*- coding: utf-8 -*-
# @Time    : 22/3/25 9:48
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : BaseNet.py
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, backbone, n_channels, num_classes):
        super(BaseModel, self).__init__()

    def get_backbone_layers(self):
        small_lr_layers = []
        return small_lr_layers

    def optim_parameters(self, lr):
        backbone_layer_id = [ii for m in self.get_backbone_layers() for ii in list(map(id, m.parameters()))]
        backbone_layer = filter(lambda p: id(p) in backbone_layer_id, self.parameters())
        rest_layer = filter(lambda p: id(p) not in backbone_layer_id, self.parameters())
        return [{'params': backbone_layer, 'lr': lr},
                {'params': rest_layer, 'lr': lr}]

    def forward(self, x):
        return x