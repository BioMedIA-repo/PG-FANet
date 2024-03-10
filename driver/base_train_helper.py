# -*- coding: utf-8 -*-
# @Time    : 19/11/8 9:54
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : base_syn_helper.py
from commons.utils import *
from tensorboardX import SummaryWriter
import torch
from models import MODELS
import matplotlib.pyplot as plt
from driver import active_transform
from torch.utils.data import DataLoader
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import label as sk_label
from active_data.InstanceDataLoader import INSTANCEDataset, load_train_vali_data, load_test_data
from active_data.constant import *
from active_data.crop_transform import argumentation_val_normal, argumentation_train, argumentation_test_normal
from driver import std, mean
from torch.cuda import empty_cache
from collections import OrderedDict
from commons.ramps import sigmoid_rampup

plt.rcParams.update({'figure.max_open_warning': 20})


class BaseTrainHelper(object):
    def __init__(self, criterions, config):
        self.criterions = criterions
        self.config = config
        # p = next(filter(lambda p: p.requires_grad, generator.parameters()))
        self.use_cuda = config.use_cuda
        # self.device = p.get_device() if self.use_cuda else None
        self.device = config.gpu if self.use_cuda else None
        if self.config.train:
            self.make_dirs()
        self.define_log()
        self.reset_model()
        self.out_put_summary()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor

    def save_checkpoint(self, state, filename='checkpoint.pth'):
        torch.save(state, filename)

    def merge_batch(self, batch):
        image_patch = [torch.unsqueeze(image, dim=0) for inst in batch for image in inst["image_patch"]]
        # orders = np.random.permutation(len(image_patch))  #fuck u!
        # image_patch = [image_patch[o] for o in orders]
        image_patch = torch.cat(image_patch, dim=0)
        image_label = [inst["image_label"] for inst in batch for _ in inst["image_patch"]]
        # image_label = [image_label[o] for o in orders]
        image_label = torch.tensor(image_label)
        image_segment = [torch.unsqueeze(image, dim=0) for inst in batch for image in inst["image_segment"]]
        # image_segment = [image_segment[o] for o in orders]
        image_segment = torch.cat(image_segment, dim=0)
        image_name = [inst["image_name"] for inst in batch for _ in inst["image_patch"]]
        # image_name = [image_name[o] for o in orders]

        return {"image_patch": image_patch,
                "image_label": image_label,
                "image_segment": image_segment,
                "image_name": image_name
                }

    def count_parameters(self, net):
        model_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def make_dirs(self):
        if not isdir(self.config.tmp_dir):
            os.makedirs(self.config.tmp_dir)
        if not isdir(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        if not isdir(self.config.save_model_path):
            os.makedirs(self.config.save_model_path)
        if not isdir(self.config.tensorboard_dir):
            os.makedirs(self.config.tensorboard_dir)
        if not isdir(self.config.submission_dir):
            os.makedirs(self.config.submission_dir)
        code_path = join(self.config.submission_dir, 'code')
        if os.path.exists(code_path):
            shutil.rmtree(code_path)
        print(os.getcwd())
        shutil.copytree('../../', code_path, ignore=shutil.ignore_patterns('.git', '__pycache__', '*log*', '*tmp*'))

    def reset_model(self):
        if self.model:
            del self.model
            empty_cache()
        self.model = MODELS[self.config.model](backbone=self.config.backbone, n_channels=self.config.classes,
                                               num_classes=self.config.classes)
        if self.use_cuda and self.model:
            self.model.to(self.equipment)
            if len(self.config.gpu_count) > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.config.gpu_count)

    def read_data(self, dataloader):
        while True:
            for batch in dataloader:
                image = batch['image_patch']
                label = batch['image_label']
                segment = batch['image_segment']
                image = image.to(self.equipment).float()
                label = label.to(self.equipment).long()
                segment = segment.to(self.equipment).long()
                yield {
                    'image_patch': image,
                    'image_label': label,
                    'image_segment': segment,
                }

    def define_log(self):
        now = datetime.now()  # current date and time
        date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        if self.config.train:
            log_s = self.config.log_file[:self.config.log_file.rfind('.txt')]
            self.log = Logger(log_s + '_' + str(date_time) + '.txt')
        else:
            self.log = Logger(join(self.config.save_dir, 'test_log_%s.txt' % (str(date_time))))
        sys.stdout = self.log

    def move_to_cuda(self):
        if self.use_cuda and self.model:
            torch.cuda.set_device(self.config.gpu)
            self.equipment = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.equipment)
            # for key in self.criterions.keys():
            #     print(key)
            #     self.criterions[key].to(self.equipment)
            if len(self.config.gpu_count) > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.config.gpu_count)
        else:
            self.equipment = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save_model_checkpoint(self, epoch, optimizer):
        save_file = join(self.config.save_model_path, 'checkpoint_epoch_%03d.pth' % (epoch))
        self.save_checkpoint({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, filename=save_file)

    def save_last_checkpoint(self, model_optimizer=None, save_model=False, iter=0):
        opti_file_path = join(self.config.save_model_path, "iter_" + str(iter) + "_last_optim.opt")
        save_model_path = join(self.config.save_model_path, "iter_" + str(iter) + "_last_model.pt")
        ema_opti_file_path = join(self.config.save_model_path, "iter_" + str(iter) + "_last_ema.pt")
        if save_model:
            torch.save(self.model.state_dict(), save_model_path)
            if hasattr(self, 'ema'):
                torch.save(self.ema, ema_opti_file_path)

        if model_optimizer is not None:
            torch.save(model_optimizer.state_dict(), opti_file_path)

    def get_last_checkpoint(self, model_optimizer=None, load_model=False, iter=0):
        if model_optimizer is not None:
            load_file = join(self.config.save_model_path, "iter_" + str(iter) + "_last_optim.opt")
        if load_model:
            load_file = join(self.config.save_model_path, "iter_" + str(iter) + "_last_model.pt")
            # load_file = join(self.config.save_model_path, "iter_0_127_best_model.pt")
        print('loaded' + load_file)
        return torch.load(load_file, map_location=('cuda:' + str(self.device)))

    def save_best_checkpoint_iter(self, model_optimizer=None, save_model=False, iter=0, epoch=0):
        opti_file_path = join(self.config.save_model_path, "iter_%s_%s_best_optim.opt" % (str(iter), str(epoch)))
        save_model_path = join(self.config.save_model_path, "iter_%s_%s_best_model.pt" % (str(iter), str(epoch)))
        ema_opti_file_path = join(self.config.save_model_path, "iter_" + str(iter) + "_best_ema.pt")
        if save_model:
            torch.save(self.model.state_dict(), save_model_path)
            if hasattr(self, 'ema'):
                torch.save(self.ema, ema_opti_file_path)
        if model_optimizer is not None:
            torch.save(model_optimizer.state_dict(), opti_file_path)

    def save_best_checkpoint(self, model_optimizer=None, save_model=False, iter=0):
        opti_file_path = join(self.config.save_model_path, "iter_" + str(iter) + "_best_optim.opt")
        save_model_path = join(self.config.save_model_path, "iter_" + str(iter) + "_best_model.pt")
        ema_opti_file_path = join(self.config.save_model_path, "iter_" + str(iter) + "_best_ema.pt")
        if save_model:
            torch.save(self.model.state_dict(), save_model_path)
            if hasattr(self, 'ema'):
                torch.save(self.ema, ema_opti_file_path)

        if model_optimizer is not None:
            torch.save(model_optimizer.state_dict(), opti_file_path)

    def get_best_checkpoint(self, model_optimizer=None, load_model=False, iter=0):
        if model_optimizer is not None:
            load_file = join(self.config.save_model_path, "iter_" + str(iter) + "_best_optim.opt")
        if load_model:
            load_file = join(self.config.save_model_path, "iter_" + str(iter) + "_best_model.pt")
            # load_file = join(self.config.save_model_path, "iter_0_127_best_model.pt")
        print('loaded' + load_file)
        return torch.load(load_file, map_location=('cuda:' + str(self.device)))

    def load_best_optim(self, optim, iter=0):
        state_dict_file = self.get_best_checkpoint(model_optimizer=True, iter=iter)
        optim.load_state_dict(state_dict_file)
        return optim

    def load_best_state(self, iter=0):
        state_dict_file = self.get_best_checkpoint(load_model=True, iter=iter)
        net_dict = self.model.state_dict()
        net_dict.update(state_dict_file)
        self.model.load_state_dict(net_dict)
        del state_dict_file
        del net_dict
        if hasattr(self, 'ema'):
            del self.ema
            ema_opti_file_path = join(self.config.save_model_path, "iter_" + str(iter) + "_best_ema.pt")
            print('load ' + ema_opti_file_path)
            self.ema = torch.load(ema_opti_file_path, map_location=('cuda:' + str(self.device)))

    def load_last_state(self, iter=0):
        state_dict_file = self.get_last_checkpoint(load_model=True, iter=iter)
        net_dict = self.model.state_dict()
        net_dict.update(state_dict_file)
        self.model.load_state_dict(net_dict)
        del state_dict_file
        del net_dict
        if self.config.ema_decay > 0:
            if hasattr(self, 'ema'):
                del self.ema
            ema_opti_file_path = join(self.config.save_model_path, "iter_" + str(iter) + "_last_ema.pt")
            print('load ema' + ema_opti_file_path)
            self.ema = torch.load(ema_opti_file_path, map_location=('cuda:' + str(self.device)))

    def out_put_summary(self):
        self.summary_writer = SummaryWriter(self.config.tensorboard_dir)

    def write_summary(self, epoch, criterions):
        for key in criterions.keys():
            self.summary_writer.add_scalar(
                key, criterions[key], epoch)

    def update_index_split(self, index_split, most_uncertain_unlabelled_index, dataset=None):
        # new index labelled and unlabelled
        index_labelled = np.hstack((index_split[INDEX_LABELLED], most_uncertain_unlabelled_index))
        index_unlabelled = np.setdiff1d(index_split[INDEX_UNLABELLED], most_uncertain_unlabelled_index)
        index_split[INDEX_LABELLED] = index_labelled
        index_split[INDEX_UNLABELLED] = index_unlabelled
        print("new labelled size %d" % (len(np.unique(index_split[INDEX_LABELLED]))))
        # print("new labelled data: \n", index_split[INDEX_LABELLED])
        print("\nnew unlabelled size %d" % (len(np.unique(index_split[INDEX_UNLABELLED]))))

        now = datetime.now()  # current date and time
        date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        save_selected_dir = join(self.config.submission_dir, "imgs_each_budget_" + str(date_time))
        lab_set = open(
            join(self.config.submission_dir, 'most_uncertain_img_%s.txt' % (str(date_time))), 'a')
        if not exists(save_selected_dir):
            makedirs(save_selected_dir)
        shutil.rmtree(save_selected_dir)
        makedirs(save_selected_dir)
        for ll in range(len(most_uncertain_unlabelled_index)):
            idx = most_uncertain_unlabelled_index[ll]
            lab_set.write(
                "%s,%s,%s,%s" % (dataset[DATA_POOL_X][idx], dataset[DATA_POOL_Y][idx], dataset[DATA_POOL_Z][idx],
                                 dataset[DATA_POOL_N][idx]))
            lab_set.write("\n")
            copy2(dataset[DATA_POOL_X][idx], save_selected_dir)
        zipDir(save_selected_dir, save_selected_dir + '.zip')
        shutil.rmtree(save_selected_dir)
        return index_split

    def adjust_training_dataset(self, data_sets, new_training_index, batch_size=20):
        train_dataset = INSTANCEDataset(data_sets, split='labelled',
                                        labelled_index=new_training_index,
                                        transform=argumentation_train(
                                            output_size=(self.config.patch_x, self.config.patch_y),
                                            aug_num=self.config.aug_num))
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  collate_fn=self.merge_batch,
                                  shuffle=True, num_workers=self.config.workers)
        return train_loader

    def get_aug_data_loader(self, default_label_size=2000,
                            train_batch_size=8,
                            test_batch_size=8, is_full_image=True, seed=666):

        data_sets, inital_split = load_train_vali_data(default_label_size, data_name=self.config.data_name,
                                                       root=self.config.data_path,
                                                       is_full_image=is_full_image, seed=seed)

        train_loader = self.adjust_training_dataset(data_sets, inital_split[INDEX_LABELLED],
                                                    batch_size=train_batch_size)
        vali_dataset = INSTANCEDataset(data_sets, split='valid',
                                       transform=argumentation_val_normal())
        vali_loader = DataLoader(vali_dataset, batch_size=1,
                                 collate_fn=self.merge_batch, shuffle=False)
        data_sets = {
            DATA_POOL_X: data_sets[DATA_POOL_X],
            DATA_POOL_Y: data_sets[DATA_POOL_Y],
            DATA_POOL_Z: data_sets[DATA_POOL_Z],
            DATA_POOL_N: data_sets[DATA_POOL_N]
        }

        return train_loader, vali_loader, data_sets, inital_split

    def get_data(self, data_set_mtl, index_split_mtl, split='unlabeled',
                 INDEX=INDEX_UNLABELLED, shuffle=False, transform=argumentation_val_normal(), num_workers=0):
        print("Calculate %s data" % (split))
        train_dataset = INSTANCEDataset(data_set_mtl, split=split,
                                        labelled_index=index_split_mtl[INDEX],
                                        transform=transform)
        unlabelled_loader = DataLoader(train_dataset, batch_size=self.config.train_seg_batch_size,
                                       collate_fn=self.merge_batch,
                                       shuffle=shuffle, num_workers=num_workers)
        return unlabelled_loader

    def get_seg_test_data_loader(self, part='A'):
        data_sets = load_test_data(data_name=self.config.data_name, root=self.config.data_path, partA=part)
        return data_sets

    def adjust_learning_rate_g(self, optimizer, i_iter, num_steps, istuning=False):
        warmup_iter = num_steps // 20
        if i_iter < warmup_iter:
            lr = self.lr_warmup(self.config.learning_rate, i_iter, warmup_iter)
        else:
            lr = self.lr_poly(self.config.learning_rate, i_iter, num_steps, 0.9)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            if istuning:
                optimizer.param_groups[1]['lr'] = lr
            else:
                optimizer.param_groups[1]['lr'] = lr * 10
                # optimizer.param_groups[1]['lr'] = lr

    def lr_poly(self, base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    def lr_warmup(self, base_lr, iter, warmup_iter):
        return base_lr * (float(iter) / warmup_iter)

    def save_vis_prob(self, images, save_dir, image_name, prob_maps, pred_labeled, label_img, ori_h, ori_w):
        grid = make_grid(images, nrow=1, padding=2)
        save_img = np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)) * std + mean
        save_img = np.clip(save_img * 255 + 0.5, 0, 255)
        visualize(save_img, join(save_dir, image_name + '_images'))
        visualize(np.expand_dims(prob_maps[1, :, :] / np.max(prob_maps[1, :, :]), -1),
                  '{:s}/{:s}_prob_inside'.format(save_dir, image_name))
        visualize(np.expand_dims(prob_maps[2, :, :] / np.max(prob_maps[2, :, :]), -1),
                  '{:s}/{:s}_prob_contour'.format(save_dir, image_name))

        # final_pred = Image.fromarray((pred_labeled * 255).astype(np.uint16))
        # final_pred.save('{:s}/{:s}_seg.png'.format(save_dir, image_name))

        # save colored objects
        pred_colored = np.zeros((ori_h, ori_w, 3))
        for k in range(1, pred_labeled.max() + 1):
            pred_colored[pred_labeled == k, :] = np.array(get_random_color())
        visualize(pred_colored, '{:s}/{:s}_seg_colored'.format(save_dir, image_name))
        pred_colored = np.zeros((ori_h, ori_w, 3))
        label_img = label_img == 1
        label_img = mlabel(label_img)
        for k in range(1, label_img.max() + 1):
            pred_colored[label_img == k, :] = np.array(get_random_color())
        visualize(pred_colored, '{:s}/{:s}_gt_colored'.format(save_dir, image_name))

    def generate_batch(self, batch):
        images = batch['image_patch'].to(self.equipment).float()
        segment = batch['image_segment'].to(self.equipment).float()
        labels = batch['image_label'].to(self.equipment).long()
        return images, segment, labels

    def split_forward(self, image, model):
        '''
        split the image image for forward process
        '''
        size = self.config.patch_x
        overlap = size // 2
        outchannel = self.config.classes
        model.eval()
        b, c, h0, w0 = image.size()

        # zero pad for border patches
        pad_h = 0
        if h0 - size > 0:
            pad_h = (size - overlap) - (h0 - size) % (size - overlap)
            tmp = torch.zeros((b, c, pad_h, w0)).to(self.equipment)
            image = torch.cat((image, tmp), dim=2)

        if w0 - size > 0:
            pad_w = (size - overlap) - (w0 - size) % (size - overlap)
            tmp = torch.zeros((b, c, h0 + pad_h, pad_w)).to(self.equipment)
            image = torch.cat((image, tmp), dim=3)

        _, c, h, w = image.size()

        output = torch.zeros((image.size(0), outchannel, h, w)).to(self.equipment)
        for i in range(0, h - overlap, size - overlap):
            r_end = i + size if i + size < h else h
            ind1_s = i + overlap // 2 if i > 0 else 0
            ind1_e = i + size - overlap // 2 if i + size < h else h
            for j in range(0, w - overlap, size - overlap):
                c_end = j + size if j + size < w else w

                image_var = image[:, :, i:r_end, j:c_end]
                with torch.no_grad():
                    output_patch, _ = model(image_var)

                ind2_s = j + overlap // 2 if j > 0 else 0
                ind2_e = j + size - overlap // 2 if j + size < w else w
                output[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:, :, ind1_s - i:ind1_e - i,
                                                             ind2_s - j:ind2_e - j]

        output = output[:, :, :h0, :w0].to(self.equipment)
        output = output.cpu().numpy()
        return output

    def prob_2_entropy(self, prob):
        """ convert probabilistic prediction maps to weighted self-information maps
        """
        n, c, h, w = prob.size()
        return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

    def fliplr(self, img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def flipvr(self, img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(2) - 1, -1, -1).long().cuda()  # N x C x H x W
        img_flip = img.index_select(2, inv_idx)
        return img_flip

    def get_current_consistency_weight(self, epoch, consistency, consistency_rampup):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return consistency * sigmoid_rampup(epoch, consistency_rampup)
