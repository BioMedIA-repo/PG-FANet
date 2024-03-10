from commons.utils import *
from tensorboardX import SummaryWriter
from torchsummaryX import summary
import cv2
from torchviz import make_dot
from models import MODELS
import torch
from driver.base_train_helper import BaseTrainHelper
import matplotlib.pyplot as plt
from driver import OPTIM
from active_data.constant import *
from torch.cuda import empty_cache
from models.EMA import EMA, MeanTeacher

plt.rcParams.update({'figure.max_open_warning': 20})


class SEGHelper(BaseTrainHelper):
    def __init__(self, criterions, config):
        super(SEGHelper, self).__init__(criterions, config)
        # self.model = model
        # self.criterions = criterions
        # self.config = config
        # # p = next(filter(lambda p: p.requires_grad, generator.parameters()))
        # self.use_cuda = config.use_cuda
        # # self.device = p.get_device() if self.use_cuda else None
        # self.device = config.gpu if self.use_cuda else None
        # if self.config.train:
        #     self.make_dirs()
        #     self.define_log()
        # self.out_put_summary()
        # self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        # self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor

    def create_mtc(self, decay=0.999):
        if hasattr(self, 'ema'):
            del self.ema
        mtc = self.create_meteacher()
        for param in mtc.parameters():
            param.detach_()
        self.ema = MeanTeacher(mtc, decay)

    def out_put_summary(self):
        from torchstat import stat
        self.summary_writer = SummaryWriter(self.config.tensorboard_dir)
        print('Model has param %.2fM' % (self.count_parameters(self.model) / 1000000.0))
        # stat(self.model, (self.config.classes, self.config.patch_x, self.config.patch_y))
        print('Done')
        # if self.config.train:
        #     print(self.model)
        # summary(self.model.cpu(),
        #         torch.zeros((1, 3, self.config.patch_x, self.config.patch_y)))
        # x = torch.zeros((1, 3, self.config.patch_x, self.config.patch_y)).requires_grad_(False)
        # prediction = self.model(x)
        # g = make_dot(prediction)
        # save_path = join(self.config.save_dir, 'graph')
        # g.render(save_path, view=False)

    def create_meteacher(self):
        mtc = MODELS[self.config.model](backbone=self.config.backbone, n_channels=3,
                                        num_classes=self.config.classes)
        mtc.to(self.device)
        return mtc

    def reset_model(self):
        if hasattr(self, 'model'):
            del self.model
            empty_cache()
        print("Creating models....")
        self.model = MODELS[self.config.model](backbone=self.config.backbone, n_channels=3,
                                               num_classes=self.config.classes)

    def generate_batch(self, batch, branch='seg'):
        images = batch['image_patch'].to(self.equipment).float()
        segment = batch['image_segment'].to(self.equipment).long()
        labels = batch['image_label'].to(self.equipment).long()
        return images, segment, labels

    def reset_optim(self):
        optimizer = OPTIM[self.config.learning_algorithm](
            self.model.optim_parameters(self.config.learning_rate),
            lr=self.config.learning_rate, weight_decay=5e-4)
        return optimizer
