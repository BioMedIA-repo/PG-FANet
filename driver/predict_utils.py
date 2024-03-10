# -*- coding: utf-8 -*-
# @Time    : 20/7/16 10:35
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : predict_utils.py
import matplotlib
import sys

sys.path.extend(["../../", "../", "./"])
from commons.utils import *
from commons.evaluation import accuracy_pixel_level, gland_accuracy_object_level, nuclei_accuracy_object_level

import torch
import imageio
from torch.cuda import empty_cache
from active_data.constant import *
import time
from driver import std, mean
from scipy.special import softmax
from commons.visualization_utils import colorize_mask
from skimage import measure
from active_data.crop_transform import argumentation_val_normal
from skimage import morphology
from torch.nn.utils import clip_grad_norm_
from module.losses import *
from active_data.lib import *


def pil_loader(path, isGT=False):
    if isGT:
        img = Image.open(path)
    else:
        img = Image.open(path).convert('RGB')
    return img


def pred_img(train_helper, image, model):
    image = image.to(train_helper.equipment).float()
    # if not (MONUSEG_DATA == train_helper.config.data_name or TNBC_DATA == train_helper.config.data_name):
    #     output1, out_list = train_helper.model(image)
    #     # for ll in range(1, len(out_list)):
    #     #     output1 += out_list[ll]
    #     output1 = output1.cpu().numpy()
    # else:
    output1 = train_helper.split_forward(image, model)
    output1 = output1.squeeze(0)

    # image = image.to(train_helper.equipment).float()
    # output1, out_list = model(image)
    # output1 = output1.cpu().numpy()
    # output1 = output1.squeeze(0)
    return output1


def valid_img(train_helper, image, model):
    image = image.to(train_helper.equipment).float()
    output1, out_list = model(image)
    output1 = output1.cpu().numpy()
    output1 = output1.squeeze(0)
    return output1, out_list


def valid_img_spilit(train_helper, image):
    output1 = train_helper.split_forward(image, train_helper.model)
    output1 = output1.squeeze(0)
    return output1, []


def valid_img_spilit2(train_helper, image, model):
    output1 = train_helper.split_forward(image, model)
    output1 = output1.squeeze(0)
    return output1, []


# def valid_img_ema(train_helper, image):
#     image = image.to(train_helper.equipment).float()
#     output1, out_list = train_helper.ema.model(image)
#     output1 = output1.cpu().numpy()
#     output1 = output1.squeeze(0)
#     return output1, out_list


def post_process(train_helper, pred):
    pred_inside = pred == 1
    if train_helper.config.data_name == MONUSEG_DATA:
        min_area = 20
        radius = 2
    elif train_helper.config.data_name == TNBC_DATA:
        min_area = 20
        radius = 2
    elif train_helper.config.data_name == GLAND_DATA:
        min_area = 100
        radius = 4
    elif train_helper.config.data_name == CRAG_DATA:
        min_area = 100
        radius = 4
    pred2 = morph.remove_small_objects(pred_inside, min_area)  # remove small object
    pred_labeled = mlabel(pred2)  # connected component labeling
    pred_labeled = morph.dilation(pred_labeled, selem=morph.selem.disk(radius))
    return pred_labeled


def train_selected_data(seg_help, train_loader, optimizer, epoch, istuning=False):
    seg_help.model.train()
    results = None
    optimizer.zero_grad()
    batch_num = int(np.ceil(len(train_loader.dataset) / float(seg_help.config.train_seg_batch_size)))
    total_iter = batch_num * seg_help.config.epochs
    loss_var_lambada = 1
    loss_seg_lambada = 1
    for i, batch in enumerate(train_loader):
        seg_help.adjust_learning_rate_g(optimizer, epoch * batch_num + i, total_iter, istuning=istuning)
        images, labels, _ = seg_help.generate_batch(batch)
        logits, out_list = seg_help.model(images)
        if labels.dim() == 4:
            labels = labels.squeeze(1)
        prob_maps = F.softmax(logits, dim=1)
        loss_SEG = seg_help.criterions['cls_loss'](logits, labels) * loss_seg_lambada
        for ll in range(1, len(out_list)):
            loss_CLS = seg_help.criterions['cls_loss'](out_list[ll], labels)
            loss_SEG += loss_CLS

        target_labeled = torch.zeros(labels.size()).long()
        for k in range(labels.size(0)):
            target_labeled[k] = torch.from_numpy(measure.label(labels[k].detach().cpu().numpy() == 1))
        target_labeled = target_labeled.cuda()
        # loss_var = seg_help.criterions['lossVar'](prob_maps, target_labeled)
        # loss = loss_SEG + loss_var * loss_var_lambada  # *(len(out_list)+1)
        loss_var = seg_help.criterions['lossVar'](prob_maps, target_labeled)
        loss = loss_SEG + loss_var * loss_var_lambada  # *(len(out_list)+1)
        loss.backward()

        # measure accuracy and record loss
        log_prob_maps = F.log_softmax(logits, dim=1)
        pred = np.argmax(log_prob_maps.detach().cpu().numpy(), axis=1)
        metrics = accuracy_pixel_level(pred, labels.detach().cpu().numpy())
        pixel_accu, iou = metrics[0], metrics[1]
        result = [loss_var.item(), loss_SEG.item(), pixel_accu, iou]
        if results is None:
            results = Averagvalue(len([*result]))
        results.update(result, images.size(0))

        # if epoch % 10 == 0 and epoch > 100:
        #     grid = make_grid(images, nrow=4, padding=2)
        #     save_img = np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)) * std + mean
        #     save_img = np.clip(save_img * 255 + 0.5, 0, 255)
        #     visualize(save_img, join(seg_help.config.tmp_dir,
        #                              str(epoch) + '_' + "images"))
        #     grid = make_grid(torch.unsqueeze(labels, dim=1), nrow=4, padding=2) / torch.max(labels).float()
        #     visualize(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)),
        #               join(seg_help.config.tmp_dir,
        #                    str(epoch) + "_label"))
        #
        #     grid = make_grid(torch.unsqueeze(prob_maps[:, 1, :, :], dim=1), nrow=4, padding=2)
        #     visualize(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)),
        #               join(seg_help.config.tmp_dir,
        #                    str(epoch) + "_0_seg"))
        #
        #     for ll in range(1, len(out_list)):
        #         prob_maps = F.softmax(out_list[ll], dim=1)
        #         grid = make_grid(torch.unsqueeze(prob_maps[:, 1, :, :], dim=1), nrow=4, padding=2)
        #         visualize(np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)),
        #                   join(seg_help.config.tmp_dir,
        #                        str(epoch) + '_' + str(ll) + "_seg"))

        if (i + 1) % seg_help.config.update_every == 0 or i == batch_num - 1:
            clip_grad_norm_(filter(lambda p: p.requires_grad, seg_help.model.parameters()), \
                            max_norm=seg_help.config.clip)
            optimizer.step()
            if hasattr(seg_help, 'ema'):
                seg_help.ema.update(seg_help.model)
            optimizer.zero_grad()
            empty_cache()
    # print("end:", torch.cuda.memory_allocated() / 1024 ** 2)
    # if (epoch + 1) % (seg_help.config.epochs // 10 if seg_help.config.epochs // 10 != 0 else 1) == 0:
    print('[Epoch {:d}/{:d}] Train Avg: [Loss Var {r[0]:.4f}]'
          ' [Loss SEG {r[1]:.4f}]'
          # ' Loss entro {r[2]:.4f}'
          ' [Acc {r[2]:.4f}]'
          ' [IoU {r[3]:.4f}]'.format(epoch, seg_help.config.epochs, r=results.avg))
    # opt_s = ''
    # for g in optimizer.param_groups:
    #     opt_s += "optimizer current_lr to %.8f \t" % (g['lr'])
    # print(opt_s)

    empty_cache()
    return {
        'train/seg_loss': results.avg[0],
        'train/seg_acc': results.avg[2],
        'train/seg_iou': results.avg[3],
    }


def train_selected_data_no_var(seg_help, train_loader, optimizer, epoch, istuning=False):
    seg_help.model.train()
    results = None
    optimizer.zero_grad()
    batch_num = int(np.ceil(len(train_loader.dataset) / float(seg_help.config.train_seg_batch_size)))
    total_iter = batch_num * seg_help.config.epochs
    loss_seg_lambada = 1
    for i, batch in enumerate(train_loader):
        seg_help.adjust_learning_rate_g(optimizer, epoch * batch_num + i, total_iter, istuning=istuning)
        images, labels, _ = seg_help.generate_batch(batch)
        logits, out_list = seg_help.model(images)
        if labels.dim() == 4:
            labels = labels.squeeze(1)
        loss_SEG = seg_help.criterions['cls_loss'](logits, labels) * loss_seg_lambada
        for ll in range(1, len(out_list)):
            loss_CLS = seg_help.criterions['cls_loss'](out_list[ll], labels)
            loss_SEG += loss_CLS

        target_labeled = torch.zeros(labels.size()).long()
        for k in range(labels.size(0)):
            target_labeled[k] = torch.from_numpy(measure.label(labels[k].detach().cpu().numpy() == 1))
        loss = loss_SEG  # *(len(out_list)+1)
        loss.backward()

        # measure accuracy and record loss
        log_prob_maps = F.log_softmax(logits, dim=1)
        pred = np.argmax(log_prob_maps.detach().cpu().numpy(), axis=1)
        metrics = accuracy_pixel_level(pred, labels.detach().cpu().numpy())
        pixel_accu, iou = metrics[0], metrics[1]
        result = [0, loss_SEG.item(), pixel_accu, iou]
        if results is None:
            results = Averagvalue(len([*result]))
        results.update(result, images.size(0))

        if (i + 1) % seg_help.config.update_every == 0 or i == batch_num - 1:
            clip_grad_norm_(filter(lambda p: p.requires_grad, seg_help.model.parameters()), \
                            max_norm=seg_help.config.clip)
            optimizer.step()
            if hasattr(seg_help, 'ema'):
                seg_help.ema.update(seg_help.model)
            optimizer.zero_grad()
            empty_cache()
    # print("end:", torch.cuda.memory_allocated() / 1024 ** 2)
    # if (epoch + 1) % (seg_help.config.epochs // 10 if seg_help.config.epochs // 10 != 0 else 1) == 0:
    print('[Epoch {:d}/{:d}] Train Avg: [Loss Var {r[0]:.4f}]'
          ' [Loss SEG {r[1]:.4f}]'
          # ' Loss entro {r[2]:.4f}'
          ' [Acc {r[2]:.4f}]'
          ' [IoU {r[3]:.4f}]'.format(epoch, seg_help.config.epochs, r=results.avg))
    # opt_s = ''
    # for g in optimizer.param_groups:
    #     opt_s += "optimizer current_lr to %.8f \t" % (g['lr'])
    # print(opt_s)

    empty_cache()
    return {
        'train/seg_loss': results.avg[0],
        'train/seg_acc': results.avg[2],
        'train/seg_iou': results.avg[3],
    }


def validtation_test2(train_helper, model2, test_loader, epoch):
    model2.eval()
    results = None
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images, labels, _ = train_helper.generate_batch(batch)
            outputs, out_list = valid_img_spilit2(train_helper, images, model2)
            # if train_helper.config.model in ['R2U_Net', 'U2NETP', 'MicroNet', 'BiONet', 'LinkNet','DCAN','MILDNet','FullNet']:
            #     outputs, out_list = valid_img_spilit(train_helper, images)
            # else:
            #     outputs, out_list = valid_img(train_helper, images)
            prob_maps = softmax(outputs, axis=0)
            pred = np.argmax(prob_maps, axis=0)

            if labels.dim() == 4:
                labels = labels.squeeze(1)
            target_labeled = torch.zeros(labels.size()).long()
            for k in range(labels.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(labels[k].detach().cpu().numpy() == 1))
            # loss_var = train_helper.criterions['lossVar'](prob_maps, target_labeled.cuda())
            # measure accuracy and record loss
            metrics = accuracy_pixel_level(np.expand_dims(pred > 0, 0), labels.detach().cpu().numpy())
            pixel_accu = metrics[0]
            result = [pixel_accu]
            if results is None:
                results = Averagvalue(len([*result]))
            results.update(result, images.size(0))
        del _
        del target_labeled
        del labels
        del images
        del prob_maps

    empty_cache()
    return {
        'vali/seg_loss': 0,
        'vali/seg_acc': results.avg[0]
    }


def validtation_test(train_helper, test_loader, epoch):
    train_helper.model.eval()
    results = None
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images, labels, _ = train_helper.generate_batch(batch)
            outputs, out_list = valid_img_spilit(train_helper, images)
            # if train_helper.config.model in ['R2U_Net', 'U2NETP', 'MicroNet', 'BiONet', 'LinkNet','DCAN','MILDNet','FullNet']:
            #     outputs, out_list = valid_img_spilit(train_helper, images)
            # else:
            #     outputs, out_list = valid_img(train_helper, images)
            prob_maps = softmax(outputs, axis=0)
            pred = np.argmax(prob_maps, axis=0)

            if labels.dim() == 4:
                labels = labels.squeeze(1)
            target_labeled = torch.zeros(labels.size()).long()
            for k in range(labels.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(labels[k].detach().cpu().numpy() == 1))
            # loss_var = train_helper.criterions['lossVar'](prob_maps, target_labeled.cuda())
            # measure accuracy and record loss
            metrics = accuracy_pixel_level(np.expand_dims(pred > 0, 0), labels.detach().cpu().numpy())
            pixel_accu = metrics[0]
            result = [pixel_accu]
            if results is None:
                results = Averagvalue(len([*result]))
            results.update(result, images.size(0))

            # image_name = batch['image_name'][0][:-4]
            # grid = make_grid(images, nrow=1, padding=2)
            # save_img = np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)) * std + mean
            # save_img = np.clip(save_img * 255 + 0.5, 0, 255)
            # visualize(save_img, join(train_helper.config.tmp_dir, image_name + '_images'))
            # grid = make_grid(torch.unsqueeze(labels, dim=1), nrow=1, padding=2)
            # save_img = grid[0].detach().cpu().numpy()
            # imgggg = colorize_mask(save_img)
            # imgggg.save(join(train_helper.config.tmp_dir, image_name + "_test_mask.png"))

            # pred_list = [pred]
            # prob_maps_list = [prob_maps]
            # for ll in range(1, len(out_list)):
            #     output1 = out_list[ll].cpu().numpy()
            #     output1 = output1.squeeze(0)
            #     prob_maps = softmax(output1, axis=0)
            #     prob_maps_list.append(prob_maps)
            #     pred = np.argmax(prob_maps, axis=0)
            #     pred_list.append(pred)
            #
            # for ll in range(len(pred_list)):
            #     prob_maps = (np.squeeze(prob_maps_list[ll]) * 255.).astype(np.uint8)
            #     imageio.imwrite(
            #         '{:s}/{:s}_{:s}_prob_inside.png'.format(train_helper.config.tmp_dir, image_name, str(ll)),
            #         prob_maps[1, :, :])
            #     imageio.imwrite(
            #         '{:s}/{:s}_{:s}_prob_contour.png'.format(train_helper.config.tmp_dir, image_name, str(ll)),
            #         prob_maps[2, :, :])

        #         pred_labeled = post_process(train_helper, pred_list[ll])
        #         pred_colored = np.zeros((images.size(2), images.size(3), 3))
        #         for k in range(1, pred_labeled.max() + 1):
        #             pred_colored[pred_labeled == k, :] = np.array(get_random_color())
        #         filename = '{:s}/{:s}_{:s}_seg_colored.png'.format(train_helper.config.tmp_dir, image_name, str(ll))
        #         imageio.imwrite(filename, (pred_colored * 255.).astype(np.uint8))

        del _
        del target_labeled
        del labels
        del images
        del prob_maps

    empty_cache()
    return {
        'vali/seg_loss': 0,
        'vali/seg_acc': results.avg[0]
    }


def validtation_test_ema(train_helper, test_loader, epoch):
    train_helper.model.eval()
    train_helper.ema.model.eval()
    results_ema = None
    results = None
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images, labels, _ = train_helper.generate_batch(batch)
            if labels.dim() == 4:
                labels = labels.squeeze(1)
            target_labeled = torch.zeros(labels.size()).long()
            for k in range(labels.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(labels[k].detach().cpu().numpy() == 1))

            outputs, out_list = valid_img_spilit(train_helper, images)
            prob_maps = softmax(outputs, axis=0)
            pred = np.argmax(prob_maps, axis=0)
            # measure accuracy and record loss
            metrics = accuracy_pixel_level(np.expand_dims(pred > 0, 0), labels.detach().cpu().numpy())
            pixel_accu, iou = metrics[0], metrics[1]
            result = [pixel_accu, iou]
            if results is None:
                results = Averagvalue(len([*result]))
            results.update(result, images.size(0))

            image_name = batch['image_name'][0][:-4]
            grid = make_grid(images, nrow=1, padding=2)
            save_img = np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)) * std + mean
            save_img = np.clip(save_img * 255 + 0.5, 0, 255)
            visualize(save_img, join(train_helper.config.tmp_dir, image_name + '_images'))
            grid = make_grid(torch.unsqueeze(labels, dim=1), nrow=1, padding=2)
            save_img = grid[0].detach().cpu().numpy()
            imgggg = colorize_mask(save_img)
            imgggg.save(join(train_helper.config.tmp_dir, image_name + "_test_mask.png"))

            pred_list = [pred]
            prob_maps_list = [prob_maps]
            for ll in range(1, len(out_list)):
                output1 = out_list[ll].cpu().numpy()
                output1 = output1.squeeze(0)
                prob_maps = softmax(output1, axis=0)
                prob_maps_list.append(prob_maps)
                pred = np.argmax(prob_maps, axis=0)
                pred_list.append(pred)

            for ll in range(len(pred_list)):
                prob_maps = (np.squeeze(prob_maps_list[ll]) * 255.).astype(np.uint8)
                imageio.imwrite(
                    '{:s}/{:s}_{:s}_prob_inside.png'.format(train_helper.config.tmp_dir, image_name, str(ll)),
                    prob_maps[1, :, :])
                imageio.imwrite(
                    '{:s}/{:s}_{:s}_prob_contour.png'.format(train_helper.config.tmp_dir, image_name, str(ll)),
                    prob_maps[2, :, :])

                # pred_labeled = post_process(train_helper, pred_list[ll])
                # pred_colored = np.zeros((images.size(2), images.size(3), 3))
                # for k in range(1, pred_labeled.max() + 1):
                #     pred_colored[pred_labeled == k, :] = np.array(get_random_color())
                # filename = '{:s}/{:s}_{:s}_seg_colored.png'.format(train_helper.config.tmp_dir, image_name, str(ll))
                # imageio.imwrite(filename, (pred_colored * 255.).astype(np.uint8))

            # ----EMA--------
            outputs_ema, out_list_ema = valid_img_spilit(train_helper, images)
            prob_maps_ema = softmax(outputs_ema, axis=0)
            pred_ema = np.argmax(prob_maps_ema, axis=0)

            metrics_ema = accuracy_pixel_level(np.expand_dims(pred_ema > 0, 0), labels.detach().cpu().numpy())
            pixel_accu_ema, iou_ema = metrics_ema[0], metrics_ema[1]
            result_ema = [pixel_accu_ema, iou_ema]
            if results_ema is None:
                results_ema = Averagvalue(len([*result_ema]))
            results_ema.update(result_ema, images.size(0))

            pred_list = [pred_ema]
            prob_maps_list = [prob_maps_ema]
            for ll in range(1, len(out_list)):
                output1 = out_list[ll].cpu().numpy()
                output1 = output1.squeeze(0)
                prob_maps = softmax(output1, axis=0)
                prob_maps_list.append(prob_maps)
                pred = np.argmax(prob_maps, axis=0)
                pred_list.append(pred)

            for ll in range(len(pred_list)):
                prob_maps = (np.squeeze(prob_maps_list[ll]) * 255.).astype(np.uint8)
                imageio.imwrite(
                    '{:s}/{:s}_{:s}_prob_inside_ema.png'.format(train_helper.config.tmp_dir, image_name, str(ll)),
                    prob_maps[1, :, :])
                imageio.imwrite(
                    '{:s}/{:s}_{:s}_prob_contour_ema.png'.format(train_helper.config.tmp_dir, image_name, str(ll)),
                    prob_maps[2, :, :])

                # pred_labeled = post_process(train_helper, pred_list[ll])
                # pred_colored = np.zeros((images.size(2), images.size(3), 3))
                # for k in range(1, pred_labeled.max() + 1):
                #     pred_colored[pred_labeled == k, :] = np.array(get_random_color())
                # filename = '{:s}/{:s}_{:s}_seg_colored_ema.png'.format(train_helper.config.tmp_dir, image_name, str(ll))
                # imageio.imwrite(filename, (pred_colored * 255.).astype(np.uint8))

        del _
        del target_labeled
        del labels
        del images
        del prob_maps

    empty_cache()
    # measure elapsed time
    # print('Epoch ' + str(epoch) + ' Student Acc {r[0]:.4f} IoU {r[1]:.4f}'.format(r=results.avg))
    # print('Epoch ' + str(epoch) + ' Teacher Acc {r[0]:.4f} IoU {r[1]:.4f}'.format(r=results_ema.avg))

    return {
        'vali/seg_loss': 0,
        'vali/seg_acc': results.avg[0],
        'vali/seg_IoU': results.avg[1],
    }


def predict_all(train_helper, dataset, ac_iter, tag='A'):
    train_helper.model.eval()
    # pred_img_func = pred_img_ema if hasattr(train_helper, 'ema') else pred_img
    pred_img_func = pred_img
    save_dir = join(train_helper.config.submission_dir, 'act_iter_final_%s_%s' % (str(ac_iter), tag))
    if not exists(save_dir):
        makedirs(save_dir)
    shutil.rmtree(save_dir)
    makedirs(save_dir)
    res_log = join(train_helper.config.save_model_path, 'test_results.txt')
    if exists(res_log):
        os.remove(res_log)
    avg_results = model_eval_valid(pred_img_func, train_helper.model, train_helper, dataset, save_dir,
                                   gland_accuracy_object_level,
                                   nuclei_accuracy_object_level)
    zipDir(save_dir, save_dir + '.zip')
    shutil.rmtree(save_dir)
    info = [avg_results.avg[ii] for ii in range(len(avg_results.avg))]
    pr = ''
    rew_log = open(res_log, 'a')
    for inf in info:
        rew_log.write("%f," % (inf))
        pr += (str(inf) + '\t')
    rew_log.write("\n")
    print(pr)
    # if not (MONUSEG_DATA == train_helper.config.data_name or TNBC_DATA == train_helper.config.data_name):
    #     print('Average of all images:\t'
    #           'pixel_accu: {r[0]:.4f}\t'
    #           'recall: {r[1]:.4f}\t'
    #           'precision: {r[2]:.4f}\t'
    #           'F1: {r[3]:.4f}\t'
    #           'dice: {r[4]:.4f}\t'
    #           'iou: {r[5]:.4f}\t'
    #           'haus: {r[6]:.4f}\t'
    #           'ohd95: {r[7]:.4f}\t'
    #           'hd95: {r[8]:.4f}'.format(r=avg_results.avg))
    # else:
    #     print('Average of all images:\t'
    #           'pixel_accu: {r[0]:.4f}\t'
    #           'recall: {r[1]:.4f}\t'
    #           'precision: {r[2]:.4f}\t'
    #           'F1: {r[3]:.4f}\t'
    #           'dice: {r[4]:.4f}\t'
    #           'iou: {r[5]:.4f}\t'
    #           'haus: {r[6]:.4f}\t'
    #           'aji: {r[7]:.4f}\t'
    #           'ohd95: {r[8]:.4f}\t'
    #           'hd95: {r[9]:.4f}'.format(r=avg_results.avg))
    return avg_results.avg[3]


def predict_all_ema(train_helper, dataset, ac_iter, tag='A'):
    train_helper.model.eval()
    train_helper.ema.model.eval()
    # pred_img_func = pred_img_ema if hasattr(train_helper, 'ema') else pred_img
    pred_img_func = pred_img
    save_dir = join(train_helper.config.submission_dir, 'act_iter_final_%s_%s' % (str(ac_iter), tag))
    if not exists(save_dir):
        makedirs(save_dir)
    shutil.rmtree(save_dir)
    makedirs(save_dir)
    res_log = join(train_helper.config.save_model_path, 'test_results.txt')
    if exists(res_log):
        os.remove(res_log)
    avg_results = model_eval_valid(pred_img_func, train_helper.model, train_helper, dataset, save_dir,
                                   gland_accuracy_object_level,
                                   nuclei_accuracy_object_level)
    zipDir(save_dir, save_dir + '.zip')
    shutil.rmtree(save_dir)
    info = [avg_results.avg[ii] for ii in range(len(avg_results.avg))]
    pr = ''
    rew_log = open(res_log, 'a')
    for inf in info:
        rew_log.write("%f," % (inf))
        pr += (str(inf) + '\t')
    rew_log.write("\n")
    print(pr)
    # if not (MONUSEG_DATA == train_helper.config.data_name or TNBC_DATA == train_helper.config.data_name):
    #     print('Average of all images:\t'
    #           'pixel_accu: {r[0]:.4f}\t'
    #           'recall: {r[1]:.4f}\t'
    #           'precision: {r[2]:.4f}\t'
    #           'F1: {r[3]:.4f}\t'
    #           'dice: {r[4]:.4f}\t'
    #           'iou: {r[5]:.4f}\t'
    #           'haus: {r[6]:.4f}\t'
    #           'ohd95: {r[7]:.4f}\t'
    #           'hd95: {r[8]:.4f}'.format(r=avg_results.avg))
    # else:
    #     print('Average of all images:\t'
    #           'pixel_accu: {r[0]:.4f}\t'
    #           'recall: {r[1]:.4f}\t'
    #           'precision: {r[2]:.4f}\t'
    #           'F1: {r[3]:.4f}\t'
    #           'dice: {r[4]:.4f}\t'
    #           'iou: {r[5]:.4f}\t'
    #           'haus: {r[6]:.4f}\t'
    #           'aji: {r[7]:.4f}\t'
    #           'ohd95: {r[8]:.4f}\t'
    #           'hd95: {r[9]:.4f}'.format(r=avg_results.avg))

    pred_img_func = pred_img
    save_dir = join(train_helper.config.submission_dir, 'act_iter_final_%s_%s_ema' % (str(ac_iter), tag))
    if not exists(save_dir):
        makedirs(save_dir)
    shutil.rmtree(save_dir)
    makedirs(save_dir)
    res_log = join(train_helper.config.save_model_path, 'test_results_ema.txt')
    if exists(res_log):
        os.remove(res_log)
    avg_results = model_eval_valid(pred_img_func, train_helper.ema.model, train_helper, dataset, save_dir,
                                   gland_accuracy_object_level,
                                   nuclei_accuracy_object_level)
    zipDir(save_dir, save_dir + '.zip')
    shutil.rmtree(save_dir)
    info = [avg_results.avg[ii] for ii in range(len(avg_results.avg))]
    pr = ''
    rew_log = open(res_log, 'a')
    for inf in info:
        rew_log.write("%f," % (inf))
        pr += (str(inf) + '\t')
    rew_log.write("\n")
    print(pr)

    # if not (MONUSEG_DATA == train_helper.config.data_name or TNBC_DATA == train_helper.config.data_name):
    #     print('EMA average of all images:\t'
    #           'pixel_accu: {r[0]:.4f}\t'
    #           'recall: {r[1]:.4f}\t'
    #           'precision: {r[2]:.4f}\t'
    #           'F1: {r[3]:.4f}\t'
    #           'dice: {r[4]:.4f}\t'
    #           'iou: {r[5]:.4f}\t'
    #           'haus: {r[6]:.4f}\t'
    #           'ohd95: {r[7]:.4f}\t'
    #           'hd95: {r[8]:.4f}'.format(r=avg_results.avg))
    # else:
    #     print('EMA average of all images:\t'
    #           'pixel_accu: {r[0]:.4f}\t'
    #           'recall: {r[1]:.4f}\t'
    #           'precision: {r[2]:.4f}\t'
    #           'F1: {r[3]:.4f}\t'
    #           'dice: {r[4]:.4f}\t'
    #           'iou: {r[5]:.4f}\t'
    #           'haus: {r[6]:.4f}\t'
    #           'aji: {r[7]:.4f}\t'
    #           'ohd95: {r[8]:.4f}\t'
    #           'hd95: {r[9]:.4f}'.format(r=avg_results.avg))

    return avg_results.avg[3]


def predict_super(seg_help, final_rest, final_rest_b, test_loader_seg, test_loader_segB, nb_acl_iter):
    print("\n-----------load last state of model -----------")
    seg_help.load_last_state(iter=nb_acl_iter)
    if CRAG_DATA == seg_help.config.data_name or TNBC_DATA == seg_help.config.data_name:
        pred_res = predict_all(seg_help, test_loader_seg, nb_acl_iter, tag='A')
        final_rest.append(pred_res)
    else:
        pred_res = predict_all(seg_help, test_loader_seg, nb_acl_iter, tag='A')
        final_rest.append(pred_res)
        pred_res_b = predict_all(seg_help, test_loader_segB, nb_acl_iter, tag='B')
        final_rest_b.append(pred_res_b)

    if seg_help.config.load_best_epoch:
        print("\n-----------load best state of model -----------")
        seg_help.load_best_state(iter=nb_acl_iter)
        if CRAG_DATA == seg_help.config.data_name or TNBC_DATA == seg_help.config.data_name:
            pred_res = predict_all(seg_help, test_loader_seg, nb_acl_iter, tag='A')
            final_rest.append(pred_res)
        else:
            pred_res = predict_all(seg_help, test_loader_seg, nb_acl_iter, tag='A')
            final_rest.append(pred_res)
            pred_res_b = predict_all(seg_help, test_loader_segB, nb_acl_iter, tag='B')
            final_rest_b.append(pred_res_b)


def predict_final(seg_help, final_rest, final_rest_b, test_loader_seg, test_loader_segB, nb_acl_iter):
    print("\n-----------load last state of model -----------")
    seg_help.load_last_state(iter=nb_acl_iter)
    if CRAG_DATA == seg_help.config.data_name or TNBC_DATA == seg_help.config.data_name:
        pred_res = predict_all_ema(seg_help, test_loader_seg, nb_acl_iter, tag='A')
        final_rest.append(pred_res)
    else:
        pred_res = predict_all_ema(seg_help, test_loader_seg, nb_acl_iter, tag='A')
        final_rest.append(pred_res)
        pred_res_b = predict_all_ema(seg_help, test_loader_segB, nb_acl_iter, tag='B')
        final_rest_b.append(pred_res_b)

    if seg_help.config.load_best_epoch:
        print("\n-----------load best state of model -----------")
        seg_help.load_best_state(iter=nb_acl_iter)
        # save computational time
        if CRAG_DATA == seg_help.config.data_name or TNBC_DATA == seg_help.config.data_name:
            pred_res = predict_all_ema(seg_help, test_loader_seg, nb_acl_iter, tag='A')
            final_rest.append(pred_res)
        else:
            pred_res = predict_all_ema(seg_help, test_loader_seg, nb_acl_iter, tag='A')
            final_rest.append(pred_res)
            pred_res_b = predict_all_ema(seg_help, test_loader_segB, nb_acl_iter, tag='B')
            final_rest_b.append(pred_res_b)


def model_eval_valid(pred_img_func, model, train_helper, dataset, save_dir, gland_accuracy_object_level,
                     nuclei_accuracy_object_level):
    avg_results = None
    transform = argumentation_val_normal()
    images = dataset['test_x']
    segments = dataset['test_y']
    labels = dataset['test_z']
    names = dataset['test_n']
    with torch.no_grad():
        for i in range(len(images)):
            img = pil_loader(images[i], isGT=False)
            segment = pil_loader(segments[i], isGT=True)
            image_list1, gt_list1 = transform(img, segment)
            output1 = pred_img_func(train_helper, image_list1, model)

            img_hf = img.transpose(Image.FLIP_LEFT_RIGHT)  # horizontal flip
            img_vf = img.transpose(Image.FLIP_TOP_BOTTOM)  # vertical flip
            img_hvf = img_hf.transpose(Image.FLIP_TOP_BOTTOM)  # horizontal and vertical flips

            input_hf, _ = transform(img_hf, segment)
            input_vf, _ = transform(img_vf, segment)
            input_hvf, _ = transform(img_hvf, segment)

            output_hf = pred_img_func(train_helper, input_hf, model)
            output_vf = pred_img_func(train_helper, input_vf, model)
            output_hvf = pred_img_func(train_helper, input_hvf, model)

            # re flip
            output_hf = np.flip(output_hf, 2)
            output_vf = np.flip(output_vf, 1)
            output_hvf = np.flip(np.flip(output_hvf, 1), 2)

            # rotation 90 and flips
            img_r90 = img.rotate(90, expand=True)
            img_r90_hf = img_r90.transpose(Image.FLIP_LEFT_RIGHT)  # horizontal flip
            img_r90_vf = img_r90.transpose(Image.FLIP_TOP_BOTTOM)  # vertical flip
            img_r90_hvf = img_r90_hf.transpose(Image.FLIP_TOP_BOTTOM)  # horizontal and vertical flips

            input_r90, _ = transform(img_r90, segment)
            input_r90_hf, _ = transform(img_r90_hf, segment)
            input_r90_vf, _ = transform(img_r90_vf, segment)
            input_r90_hvf, _ = transform(img_r90_hvf, segment)

            output_r90 = pred_img_func(train_helper, input_r90, model)
            output_r90_hf = pred_img_func(train_helper, input_r90_hf, model)
            output_r90_vf = pred_img_func(train_helper, input_r90_vf, model)
            output_r90_hvf = pred_img_func(train_helper, input_r90_hvf, model)

            # re flip
            output_r90 = np.rot90(output_r90, k=3, axes=(1, 2))
            output_r90_hf = np.rot90(np.flip(output_r90_hf, 2), k=3, axes=(1, 2))
            output_r90_vf = np.rot90(np.flip(output_r90_vf, 1), k=3, axes=(1, 2))
            output_r90_hvf = np.rot90(np.flip(np.flip(output_r90_hvf, 1), 2), k=3, axes=(1, 2))

            output = (output1 + output_hf + output_vf + output_hvf
                      + output_r90 + output_r90_hf + output_r90_vf + output_r90_hvf) / 8

            ori_h = image_list1.size(2)
            ori_w = image_list1.size(3)
            label_img = gt_list1.detach().cpu().numpy()
            label_img = np.squeeze(label_img)
            image_name = names[i][:-4]
            prob_maps = softmax(output, axis=0)

            pred = np.argmax(output, axis=0)  # prediction
            pred_labeled = post_process(train_helper, pred)
            # print('\tComputing metrics...')
            result = accuracy_pixel_level(np.expand_dims(pred_labeled > 0, 0), np.expand_dims(label_img > 0, 0))
            pixel_accu = result[0]
            hd_95 = result[1]

            if MONUSEG_DATA == train_helper.config.data_name:
                result_object = nuclei_accuracy_object_level(pred_labeled, label_img)
            elif TNBC_DATA == train_helper.config.data_name:
                result_object = nuclei_accuracy_object_level(pred_labeled, label_img)
                label_img[label_img > 0] = 1
            elif CRAG_DATA == train_helper.config.data_name:
                result_object = gland_accuracy_object_level(pred_labeled, label_img)
                label_img[label_img > 0] = 1
            elif GLAND_DATA == train_helper.config.data_name:
                result_object = gland_accuracy_object_level(pred_labeled, label_img)

            if avg_results is None:
                avg_results = Averagvalue(len([pixel_accu, *result_object, hd_95]))
            avg_results.update([pixel_accu, *result_object, hd_95])
            train_helper.save_vis_prob(image_list1, save_dir, image_name, prob_maps, pred_labeled, label_img, ori_h,
                                       ori_w)
    return avg_results
