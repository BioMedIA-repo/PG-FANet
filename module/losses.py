import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss
import numpy as np


class LossVariance(nn.Module):
    """ The instances in target should be labeled
    """

    def __init__(self):
        super(LossVariance, self).__init__()

    def forward(self, input, target):
        B = input.size(0)

        loss = torch.zeros(1).float().to(input.get_device())
        for k in range(B):
            unique_vals = target[k].unique()
            unique_vals = unique_vals[unique_vals != 0]

            sum_var = torch.zeros(1).float().to(input.get_device())
            for val in unique_vals:
                instance = input[k][:, target[k] == val]
                if instance.size(1) > 1:
                    sum_var += instance.var(dim=1).sum()

            loss += sum_var / (len(unique_vals) + 1e-8)
            # print(len(unique_vals))
        loss /= B
        # print(loss)
        # print(B)
        return loss


def make_one_hot(labels, classes):
    one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target


class WeightedSoftDiceLoss(nn.Module):
    '''
    from
    https://kaggle2.blob.core.windows.net/forum-message-attachments/212369/7045/weighted_dice_2.png
    '''

    def __init__(self):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, inputs, targets, weights):
        num = targets.size(0)
        m1 = inputs.view(num, -1)
        m2 = targets.view(num, -1)
        w = weights.view(num, -1)
        w2 = w * w
        intersection = (m1 * m2)

        score = 2. * ((w2 * intersection).sum(1) + 1) / ((w2 * m1).sum(1) + (w2 * m2).sum(1) + 1)
        score = 1 - score.sum() / num
        return max(score, 0)


class DC_and_Focal_loss(nn.Module):
    def __init__(self, gamma=0.25):
        super(DC_and_Focal_loss, self).__init__()
        self.fl = FocalLoss(gamma=gamma)
        self.dc = DiceLoss()

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        fl_loss = self.fl(net_output, target)
        result = fl_loss + dc_loss
        return result


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):
        num = targets.size(0)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return max(score, 0)


class DC_and_BCE_loss(nn.Module):
    def __init__(self, ):
        super(DC_and_BCE_loss, self).__init__()
        self.ce = BCELoss()
        self.dc = SoftDiceLoss()

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        result = ce_loss + dc_loss
        return result


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss


class EntropyLoss(nn.Module):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """

    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, v):
        assert v.dim() == 4
        n, c, h, w = v.size()
        return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))



def softmax_aug_etp_lr_mse_loss(input_logits, input_logits2, target_logits):
    assert input_logits.size() == target_logits.size()
    shape_aware = torch.abs(F.softmax(input_logits, dim=1) - F.softmax(input_logits2, dim=1))
    entr_loss = entropy_loss(shape_aware, C=2)
    # shape_entropy = -1.0 * torch.sum(shape_aware * torch.log(shape_aware + 1e-12), dim=1, keepdim=True)
    # entr_loss = torch.mean(shape_entropy)
    #     mask = (att == 0).float()
    #     maksed_weight = att + mask
    #     entropy_loss = -att * torch.log(maksed_weight)
    #     entropy_loss = entropy_loss.sum() / labeled_imgs.size(0)

    input_softmax = F.softmax(input_logits, dim=1)
    input_softmax2 = F.softmax(input_logits2, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    tea_uncert = - (target_softmax * torch.log(target_softmax + 1e-12))
    tt = (1 - tea_uncert) * target_softmax + tea_uncert * input_softmax
    mse_loss = ((input_softmax - tt) ** 2 + (input_softmax - input_softmax2) ** 2) + entr_loss
    return mse_loss


def softmax_aug_msketp_lr_mse_loss(input_logits, input_logits2, target_logits):
    assert input_logits.size() == target_logits.size()
    shape_aware = torch.abs(F.softmax(input_logits, dim=1) - F.softmax(target_logits, dim=1))
    shape_entropy = -1.0 * torch.sum(shape_aware * torch.log(shape_aware + 1e-12), dim=1, keepdim=True) / torch.tensor(
        np.log(2))
    max_val, _ = torch.max(shape_entropy.view(shape_entropy.size(0), -1), dim=1, keepdim=True)
    min_val, _ = torch.min(shape_entropy.view(shape_entropy.size(0), -1), dim=1, keepdim=True)
    max_val = max_val.view(shape_entropy.size(0), 1, 1, 1).expand_as(shape_entropy)
    min_val = min_val.view(shape_entropy.size(0), 1, 1, 1).expand_as(shape_entropy)
    att_mask = (shape_entropy - min_val) / (max_val - min_val)

    # entr_loss = torch.mean(shape_entropy)
    #     mask = (att == 0).float()
    #     maksed_weight = att + mask
    #     entropy_loss = -att * torch.log(maksed_weight)
    #     entropy_loss = entropy_loss.sum() / labeled_imgs.size(0)

    # attention = torch.mean(x, dim=1, keepdim=True)
    # max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
    # threshold = max_val * np.random.uniform(0.7, 0.9)
    # threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    # drop_mask = (attention < threshold).float()
    # return x.mul(drop_mask)

    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    tea_uncert = - (target_softmax * torch.log(target_softmax + 1e-12))
    tt = (1 - tea_uncert) * target_softmax + tea_uncert * input_softmax

    input_softmax2 = F.softmax(input_logits2, dim=1)
    # mse_loss = att_mask * (input_softmax - tt) ** 2 + (input_softmax - input_softmax2) ** 2
    mse_loss = (1 + att_mask) * (input_softmax - tt) ** 2 + (input_softmax - input_softmax2) ** 2
    return mse_loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / \
         torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

