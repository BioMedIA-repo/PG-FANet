import matplotlib
import sys

sys.path.extend(["../../", "../", "./"])
from commons.utils import *
import torch
from torch.cuda import empty_cache
import random
from active_data.constant import *
from driver.seg_driver.SEGHelper import SEGHelper
from driver.Config import Configurable
from driver.predict_utils import *
from module.losses import EntropyLoss, CE_DiceLoss
import pickle
from torch.nn import CrossEntropyLoss
from shutil import copy2
from active_data.crop_transform import argumentation_train

matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()
import argparse

log_template = "[Epoch %d/%d] [seg loss: %f] [seg acc: %f]"


def main(config, seed=666, args=None):
    criterion = {
        # 'seg_loss': DiceLoss(),
        # 'loss': DC_and_Focal_loss(),
        'cls_loss': CE_DiceLoss(),
        # 'loss': CE_DiceLoss(),
        'entro': EntropyLoss(),
        'lossVar': LossVariance(),
    }
    seg_help = SEGHelper(criterion,
                         config)
    seg_help.move_to_cuda()
    optimizer = seg_help.reset_optim()
    print("data name ", seg_help.config.data_name)
    print("data size ", seg_help.config.patch_x)
    print("Random dataset Seed: %d" % (seed))
    if args:
        print('--inter %s, --intra %s, --shape %s --KL %s, --consis %s, --random-noise %s, --mask %s' % (
            str(args.inter), str(args.intra), str(args.shape), str(args.KL), str(args.consis), str(args.random_noise),
            str(args.mask)))
    train_loader_acdrl, \
    vali_loader_acdrl, \
    data_sets_acdrl, \
    index_split_acdrl = seg_help.get_aug_data_loader(
        default_label_size=seg_help.config.default_seg_label_size,
        train_batch_size=seg_help.config.train_seg_batch_size,
        test_batch_size=seg_help.config.test_seg_batch_size, is_full_image=False, seed=seed)

    # unlabelled_loader = seg_help.get_data(data_sets_acdrl,
    #                                       index_split_acdrl, split='unlabeled',
    #                                       shuffle=True, num_workers=seg_help.config.workers)

    unlabelled_loader = seg_help.get_data(data_sets_acdrl,
                                          index_split_acdrl, split='unlabeled',
                                          shuffle=True, num_workers=seg_help.config.workers,
                                          transform=argumentation_train(
                                              output_size=(seg_help.config.patch_x, seg_help.config.patch_y),
                                              aug_num=seg_help.config.aug_num))

    test_loader_seg = seg_help.get_seg_test_data_loader()
    final_rest = []
    final_rest_b = []
    test_loader_segB = []
    if CRAG_DATA != seg_help.config.data_name:
        test_loader_segB = seg_help.get_seg_test_data_loader(part='B')
        final_rest_b = []
    start_acl_iter = 0

    # --------------resume training----------------------
    for nb_acl_iter in range(start_acl_iter, seg_help.config.nb_active_learning_iter):
        best_acc = 0
        bad_step = 0
        seg_help.create_mtc(decay=seg_help.config.ema_decay)
        try:
            seg_help.load_last_state(iter=nb_acl_iter)
        except Exception as er:
            print(er)

        for epoch in range(seg_help.config.epochs):
            train_semi_mse(seg_help, train_loader_acdrl, unlabelled_loader, optimizer, epoch,
                           istuning=False, args=args)
            # validtation_test_func = validtation_test_ema if hasattr(seg_help, 'ema') else validtation_test
            seg_help.save_last_checkpoint(model_optimizer=optimizer, save_model=True, iter=nb_acl_iter)
            seg_help.log.flush()
            if (epoch + 1) % 1 == 0:
                vali_critics = validtation_test(seg_help, vali_loader_acdrl, epoch)
                if vali_critics['vali/seg_acc'] >= best_acc:
                    # st = 'EMA' if hasattr(seg_help,+ 'ema') else ''
                    print(" * Best vali acc at epoch %d: history = %.4f, current = %.4f" % (epoch, best_acc,
                                                                                            vali_critics[
                                                                                                'vali/seg_acc']))
                    best_acc = vali_critics['vali/seg_acc']
                    seg_help.write_summary(nb_acl_iter, vali_critics)
                    seg_help.save_best_checkpoint(model_optimizer=optimizer, save_model=True, iter=nb_acl_iter)
                    opt_s = ''
                    for g in optimizer.param_groups:
                        opt_s += "optimizer current_lr to %.8f \t" % (g['lr'])
                    print(opt_s)
                    bad_step = 0
                else:
                    bad_step += 1
                    if bad_step >= seg_help.config.patience:
                        seg_help.save_last_checkpoint(model_optimizer=optimizer, save_model=True, iter=nb_acl_iter)
                        # break

        predict_final(seg_help, final_rest, final_rest_b, test_loader_seg, test_loader_segB, nb_acl_iter)

        empty_cache()
        seg_help.log.flush()
        seg_help.reset_model()
        seg_help.move_to_cuda()
        optimizer = seg_help.reset_optim()

    save_dir = seg_help.config.save_dir + '_A.pkl'
    pickle.dump(final_rest, open(save_dir, 'wb'))
    if not (CRAG_DATA == seg_help.config.data_name or TNBC_DATA == seg_help.config.data_name):
        save_dir = seg_help.config.save_dir + '_B.pkl'
        pickle.dump(final_rest_b, open(save_dir, 'wb'))
    seg_help.summary_writer.close()


def train_semi_mse(seg_help, train_loader, unlabelled_loader, optimizer, epoch, istuning=False, args=None):
    def read_data(train_helper, dataloader):
        while True:
            for batch in dataloader:
                images, labels, _ = train_helper.generate_batch(batch)
                yield images, labels, _

    seg_help.model.train()
    seg_help.ema.model.train()
    results = None
    optimizer.zero_grad()
    batch_num = int(np.ceil(
        (len(train_loader.dataset) + len(unlabelled_loader.dataset)) // 2 / float(
            seg_help.config.train_seg_batch_size)))
    total_iter = batch_num * seg_help.config.epochs
    loss_var_lambada = 1
    loss_seg_lambada = 1
    labeled_train_iter = read_data(seg_help, train_loader)
    unlabeled_train_iter = read_data(seg_help, unlabelled_loader)

    for i in range(batch_num):
        seg_help.adjust_learning_rate_g(optimizer, epoch * batch_num + i, total_iter, istuning=istuning)
        images, labels, _ = next(labeled_train_iter)
        images_un, labels_un, _ = next(unlabeled_train_iter)
        images_un_noise = transforms_for_noise(images_un, 0.5, seg_help.device)
        if random.random() > 0.5:
            logits_un, out_list_un = seg_help.model(images_un_noise, has_dropout=True, random_noise=True)
            with torch.no_grad():
                logits_un_ema, out_list_un_ema = seg_help.ema.model(images_un)
        else:
            logits_un, out_list_un = seg_help.model(images_un, has_dropout=True, random_noise=True)
            with torch.no_grad():
                logits_un_ema, out_list_un_ema = seg_help.ema.model(images_un_noise)
        # S7
        # consistency_dist = softmax_mse_loss(logits_un, logits_un_ema)
        # consistency_dist = softmax_inter_extreme_lr_mse_loss(logits_un, out_list_un[1], logits_un_ema)
        # consistency_dist = softmax_extreme_lr_mse_loss(logits_un, out_list_un[1], logits_un_ema)
        if args.mask:
            # Inter T, Intra T, Shape T
            consistency_dist = softmax_aug_msketp_lr_mse_loss(logits_un, out_list_un[1], logits_un_ema)
        else:
            # Inter T, Intra T, Shape T
            consistency_dist = softmax_aug_etp_lr_mse_loss(logits_un, out_list_un[1], logits_un_ema)

        consistency_dist = torch.mean(consistency_dist)

        consistency_dist_value = consistency_dist.item()
        # 100过于大了
        consistency_weight = seg_help.get_current_consistency_weight(epoch, float(args.consis), seg_help.config.epochs)
        if labels.dim() == 4:
            labels = labels.squeeze(1)
        if labels_un.dim() == 4:
            labels_un = labels_un.squeeze(1)

        logits, out_list = seg_help.model(images)
        prob_maps = F.softmax(logits, dim=1)
        loss_SEG = seg_help.criterions['cls_loss'](logits, labels) * loss_seg_lambada
        for ll in range(1, len(out_list)):
            loss_CLS = seg_help.criterions['cls_loss'](out_list[ll], labels)
            loss_SEG += loss_CLS

        target_labeled = torch.zeros(labels.size()).long()
        for k in range(labels.size(0)):
            target_labeled[k] = torch.from_numpy(measure.label(labels[k].detach().cpu().numpy() == 1))
        target_labeled = target_labeled.cuda()
        loss_var = seg_help.criterions['lossVar'](prob_maps, target_labeled)
        loss = loss_SEG + loss_var * loss_var_lambada + consistency_dist * consistency_weight  # *(len(out_list)+1)
        loss.backward()

        # measure accuracy and record loss
        log_prob_maps = F.log_softmax(logits, dim=1)
        pred = np.argmax(log_prob_maps.detach().cpu().numpy(), axis=1)
        metrics = accuracy_pixel_level(pred, labels.detach().cpu().numpy())
        pixel_accu, iou = metrics[0], metrics[1]
        result = [loss_var.item(), loss_SEG.item(), pixel_accu, iou, consistency_dist_value,
                  consistency_weight, consistency_dist_value * consistency_weight]
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
    # if epoch % (seg_help.config.epochs // 50 if seg_help.config.epochs // 50 != 0 else 1) == 0:
    print('[Epoch {:d}/{:d}] Train Avg: [Loss Var {r[0]:.4f}]'
          ' [Loss SEG {r[1]:.4f}]'
          ' [Acc {r[2]:.4f}]'
          ' [IoU {r[3]:.4f}]'
          ' [Consist {r[4]:.4f}]'
          ' [WConsist {r[5]:.4f}]'
          ' [WC {r[6]:.4f}]'
          .format(epoch, seg_help.config.epochs, r=results.avg))
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


if __name__ == '__main__':
    torch.manual_seed(6666)
    torch.cuda.manual_seed(6666)
    random.seed(6666)
    np.random.seed(6666)
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    torch.backends.cudnn.benchmark = True  # cudn
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config-file', default='seg_configuration.txt')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--train', help='test not need write', default=True)
    argparser.add_argument('--gpu', help='GPU 0,1,2,3', default=0)
    argparser.add_argument('--gpu-count', help='number of GPUs (0,1,2,3)', default='0')
    argparser.add_argument('--run-num', help='run num: 0,2,3', default=0)
    argparser.add_argument('--active-iter', help='active iter', default="1")
    argparser.add_argument('--ema-decay', help='ema decay', default=0.99)
    argparser.add_argument('--default-size', help='default size', default="170")
    argparser.add_argument('--seed', help='random seed', default=666, type=int)
    argparser.add_argument('--model', help='model name', default="UNet")
    argparser.add_argument('--inter', type=str2bool, default='False')
    argparser.add_argument('--intra', type=str2bool, default='False')
    argparser.add_argument('--shape', type=str2bool, default='False')
    argparser.add_argument('--KL', type=str2bool, default='False')
    argparser.add_argument('--random-noise', type=str2bool, default='False')
    argparser.add_argument('--mask', type=str2bool, default='False')
    argparser.add_argument('--consis', help='consistency', default=0.1)
    args, extra_args = argparser.parse_known_args()
    config = Configurable(args, extra_args)
    torch.set_num_threads(config.workers + 1)
    print('--inter %s, --intra %s, --shape %s --KL %s, --consis %s, --random-noise %s, --mask %s' % (
        str(args.inter), str(args.intra), str(args.shape), str(args.KL), str(args.consis), str(args.random_noise),
        str(args.mask)))
    config.train = args.train
    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)
    main(config, seed=args.seed, args=args)
