from commons.utils import *
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import compute_class_weight as sk_compute_class_weight
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import f1_score, average_precision_score
from scipy.spatial.distance import directed_hausdorff as hausdorff
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import torch

from medpy.metric.binary import hd95



def evaluate_acc_jaccard(cm):
    # Compute metrics
    TP_perclass = cm.diagonal().astype('float32')
    jaccard_perclass = TP_perclass / (cm.sum(1) + cm.sum(0) - TP_perclass)
    jaccard = np.mean(jaccard_perclass)
    accuracy = TP_perclass.sum() / cm.sum()

    return accuracy, jaccard, jaccard_perclass




def calculate_metrics(y_test, y_pred):
    """Calculates the accuracy metrics"""

    accuracy = accuracy_score(y_test, y_pred)

    # Wrapping all the scoring function calls in a try & except to prevent
    # the following warning to result in a "TypeError: warnings_to_log()
    # takes 4 positional arguments but 6 were given" when sklearn calls
    # warnings.warn with an "UndefinedMetricWarning:Precision is
    # ill-defined and being set to 0.0 in labels with no predicted
    # samples." message on python 3.7.x
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)

    return [accuracy, precision, recall, f1score]


def compute_class_weights(y, wt_type='balanced', return_dict=True):
    # need to check if y is one hot
    if len(y.shape) > 1:
        y = y.argmax(axis=-1)

    assert wt_type in ['ones', 'balanced', 'balanced-sqrt'], 'Weight type not supported'

    classes = np.unique(y)
    class_weights = np.ones(shape=classes.shape[0])

    if wt_type == 'balanced' or wt_type == 'balanced-sqrt':

        class_weights = sk_compute_class_weight(class_weight='balanced',
                                                classes=classes,
                                                y=y)
        if wt_type == 'balanced-sqrt':
            class_weights = np.sqrt(class_weights)

    if return_dict:
        class_weights = dict([(i, w) for i, w in enumerate(class_weights)])

    return class_weights


def jaccard(y_true, y_pred):
    intersect = np.sum(y_true * y_pred)  # Intersection points
    union = np.sum(y_true) + np.sum(y_pred)  # Union points
    return (float(intersect)) / (union - intersect + 1e-7)


def dice_coef(y_true, y_pred):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(y_true).astype(np.bool)
    im2 = np.asarray(y_pred).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def compute_all_metric_for_seg(y_true, y_pred):
    batch, channel, width, height = y_true.shape
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    fpr, tpr, thresholds = roc_curve((y_true), y_pred)
    AUC_ROC = roc_auc_score(y_true, y_pred)
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
    # roc_curve = plt.figure()
    # plt.plot(fpr, tpr, '-', label=algorithm + '_' + dataset + '(AUC = %0.4f)' % AUC_ROC)
    # plt.title('ROC curve', fontsize=14)
    # plt.xlabel("FPR (False Positive Rate)", fontsize=15)
    # plt.ylabel("TPR (True Positive Rate)", fontsize=15)
    # plt.legend(loc="lower right")
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    confusion = confusion_matrix(y_true, y_pred)
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    mean_jaccard, thresholded_jaccard = compute_jaccard(np.reshape(y_true, (batch, width, height)),
                                                        np.reshape(y_pred, (batch, width, height)))
    specificity = 0
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    sensitivity = 0
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    precision = 0
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    dice_score = dice_coef(y_true, y_pred)
    print("Area under the ROC curve: " + str(AUC_ROC)
          + "\nArea under Precision-Recall curve: " + str(AUC_prec_rec)
          + "\nMean Jaccard similarity score: " + str(mean_jaccard)
          + "\nF1 score (F-measure): " + str(F1_score)
          + "\nACCURACY: " + str(accuracy)
          + "\nSENSITIVITY: " + str(sensitivity)
          + "\nSPECIFICITY: " + str(specificity)
          + "\nPRECISION: " + str(precision)
          + "\nDICE SCORE: " + str(dice_score)
          )


def compute_all_metric_for_single_seg(y_true, y_pred):
    tensor_y_pred = torch.from_numpy(y_pred).cuda().float()
    tensor_y_true = torch.from_numpy(y_true).cuda().float()
    accuracy = get_accuracy(tensor_y_pred, tensor_y_true)
    sensitivity = get_sensitivity(tensor_y_pred, tensor_y_true)
    specificity = get_specificity(tensor_y_pred, tensor_y_true)
    dice_score = get_DC(tensor_y_pred, tensor_y_true)
    mean_jaccard = get_JS(tensor_y_pred, tensor_y_true)
    F1_score = get_F1(tensor_y_pred, tensor_y_true)
    scores = {  # 'ROC': [],
        # 'Precision-Recall': [],
        'Jaccard': [],
        'F1': [], 'ACCURACY': [], 'SENSITIVITY': [], 'SPECIFICITY': [],
        # 'PRECISION': [],
        # 'DICEDIST': [],
        'DICESCORE': []}
    # scores['ROC'].append(AUC_ROC)
    scores['Jaccard'].append(mean_jaccard)
    scores['F1'].append(F1_score)
    scores['ACCURACY'].append(accuracy)
    scores['SENSITIVITY'].append(sensitivity)
    scores['SPECIFICITY'].append(specificity)
    # scores['PRECISION'].append(precision)
    scores['DICESCORE'].append(dice_score)
    return scores


def calculate_cls_metric(y_true, y_pred, task=1, thres=0.5):
    if len(np.unique(y_true)) == 3:
        task1_output_map = lambda x: 1 if x == 0 else 0
        task2_output_map = lambda x: 1 if x == 1 else 0
        task_output_map = task1_output_map if task == 1 else task2_output_map
        labels_task = list(map(task_output_map, y_true))
        AUC_ROC = roc_auc_score(labels_task, y_pred[:, task - 1])
        # preds_task = list(map(task_output_map, y_pred_hard))
        accuracy = accuracy_score(
            labels_task, np.where(y_pred[:, task - 1] >= thres, 1, 0))
        precision = average_precision_score(labels_task, y_pred[:, task - 1])
        conf_matrix = confusion_matrix(labels_task, y_pred[:, task - 1] >= thres)

    elif len(np.unique(y_true)) == 2:
        labels_task = y_true
        # preds_task = y_pred[:, task] >= 0.5
        AUC_ROC = roc_auc_score(labels_task, y_pred[:, 1])
        accuracy = accuracy_score(
            labels_task, np.where(y_pred[:, 1] >= thres, 1, 0))
        precision = average_precision_score(labels_task, y_pred[:, 1])
        conf_matrix = confusion_matrix(labels_task, y_pred[:, 1] >= thres)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    # confusion = confusion_matrix(labels_task, preds_task)
    # accuracy = 0
    # if float(np.sum(confusion)) != 0:
    #     accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    # specificity = 0
    # if float(confusion[0, 0] + confusion[0, 1]) != 0:
    #     specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    # sensitivity = 0
    # if float(confusion[1, 1] + confusion[1, 0]) != 0:
    #     sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    # precision = 0
    # if float(confusion[1, 1] + confusion[0, 1]) != 0:
    #     precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    print("----------Task : " + str(task) + " Metric----------"
          + "\nAUC: " + str(AUC_ROC)
          + "\nACCURACY: " + str(accuracy)
          + "\nSENSITIVITY: " + str(sensitivity)
          + "\nSPECIFICITY: " + str(specificity)
          + "\nPRECISION: " + str(precision)
          )
    return AUC_ROC, accuracy, sensitivity, specificity, precision


def compute_all_metric_for_class_wise_cls(y_true, y_pred):
    labels_task = y_true
    preds_task = y_pred
    lesion_cls_metrics = {}
    AUC_ROC, accuracy, sensitivity, specificity, precision = calculate_cls_metric(labels_task, preds_task, task=1)
    lesion_cls_metrics['task1'] = {
        'AUC': AUC_ROC,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision
    }
    AUC_ROC, accuracy, sensitivity, specificity, precision = calculate_cls_metric(labels_task, preds_task, task=2)
    lesion_cls_metrics['task2'] = {
        'AUC': AUC_ROC,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision
    }
    return lesion_cls_metrics


def compute_jaccard(y_true, y_pred):
    mean_jaccard = 0.
    thresholded_jaccard = 0.

    for im_index in range(y_pred.shape[0]):
        current_jaccard = jaccard(y_true=y_true[im_index], y_pred=y_pred[im_index])

        mean_jaccard += current_jaccard
        thresholded_jaccard += 0 if current_jaccard < 0.65 else current_jaccard

    mean_jaccard = mean_jaccard / y_pred.shape[0]
    thresholded_jaccard = thresholded_jaccard / y_pred.shape[0]

    return mean_jaccard, thresholded_jaccard


def get_accuracy(SR, GT, threshold=0.5):
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()
    corr = torch.sum(SR == GT)
    tensor_size = torch.prod(torch.tensor(SR.size()))
    acc = float(corr) / float(tensor_size)
    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    # TP : True Positive
    # FN : False Negative
    TP = (((SR == 1).int() + (GT == 1).int()).int() == 2).int()
    FN = (((SR == 0).int() + (GT == 1).int()).int() == 2).int()

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    # TN : True Negative
    # FP : False Positive
    TN = (((SR == 0).int() + (GT == 0).int()).int() == 2).int()
    FP = (((SR == 1).int() + (GT == 0).int()).int() == 2).int()

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    # TP : True Positive
    # FP : False Positive
    TP = (((SR == 1).int() + (GT == 1).int()).int() == 2).int()
    FP = (((SR == 1).int() + (GT == 0).int()).int() == 2).int()

    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    Inter = torch.sum((SR + GT) == 2).int()
    Union = torch.sum((SR + GT) >= 1).int()

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = (SR > threshold).int()
    GT = (GT == torch.max(GT)).int()

    Inter = torch.sum((SR + GT) == 2).int()
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC


def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.

    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.

    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return out_classes, correct.item()


def accuracy_check(mask, prediction):
    ims = [mask, prediction]
    np_ims = []
    for item in ims:
        if 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.cpu().detach().numpy()
        np_ims.append(item)
    compare = np.equal(np.where(np_ims[0] > 0.5, 1, 0), np_ims[1])
    accuracy = np.sum(compare)
    return accuracy / len(np_ims[0].flatten())


def accuracy_pixel_level(output, target):
    """ Computes the accuracy during training and validation for ternary label """
    batch_size = target.shape[0]
    results = np.zeros((7,), np.float)

    for i in range(batch_size):
        pred = output[i, :, :]
        label = target[i, :, :]

        # inside part
        pred_inside = pred == 1
        label_inside = label == 1
        metrics_inside = compute_pixel_level_metrics(pred_inside, label_inside)

        results += np.array(metrics_inside)

    return [value / batch_size for value in results]


def compute_pixel_level_metrics(pred, target):
    """ Compute the pixel-level tp, fp, tn, fn between
    predicted img and groundtruth target
    """

    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(target, np.ndarray):
        target = np.array(target)

    tp = np.sum(pred * target)  # true postives
    tn = np.sum((1 - pred) * (1 - target))  # true negatives
    fp = np.sum(pred * (1 - target))  # false postives
    fn = np.sum((1 - pred) * target)  # false negatives

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    F1 = 2 * precision * recall / (precision + recall + 1e-10)
    acc = (tp + tn) / (tp + fp + tn + fn + 1e-10)
    performance = (recall + tn / (tn + fp + 1e-10)) / 2
    iou = tp / (tp + fp + fn + 1e-10)
    hd_95 = hd95(pred, target) if tp > 0 else 0
    return [acc, hd_95, iou, recall, precision, F1, performance]


def nuclei_accuracy_object_level(pred, gt):
    """ Computes the accuracy during test phase of nuclei segmentation """
    # get connected components
    pred_labeled = mlabel(pred)
    gt_labeled = mlabel(gt)
    Ns = len(np.unique(pred_labeled)) - 1  # number of detected objects
    Ng = len(np.unique(gt_labeled)) - 1  # number of ground truth objects

    TP = 0.0  # true positive
    FN = 0.0  # false negative
    dice = 0.0
    haus = 0.0
    hd_95 = 0.0
    iou = 0.0
    C = 0.0
    U = 0.0
    # pred_copy = np.copy(pred)
    count = 0.0

    for i in range(1, Ng + 1):
        gt_i = np.where(gt_labeled == i, 1, 0)
        overlap_part = pred_labeled * gt_i

        # get intersection objects numbers in pred_labeled
        obj_no = np.unique(overlap_part)
        obj_no = obj_no[obj_no != 0]

        # no intersection object
        if obj_no.size == 0:
            FN += 1
            U += np.sum(gt_i)
            continue

        # find max iou object
        max_iou = 0.0
        for k in obj_no:
            tmp_overlap_area = np.sum(overlap_part == k)
            tmp_pred = np.where(pred_labeled == k, 1, 0)  # segmented object
            tmp_iou = float(tmp_overlap_area) / (np.sum(tmp_pred) + np.sum(gt_i) - tmp_overlap_area)
            if tmp_iou > max_iou:
                max_iou = tmp_iou
                pred_i = tmp_pred
                overlap_area = tmp_overlap_area

        TP += 1
        count += 1

        # compute dice and iou
        dice += 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
        iou += float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)

        # compute hausdorff distance
        seg_ind = np.argwhere(pred_i)
        gt_ind = np.argwhere(gt_i)
        haus += max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])
        hd_95 += hd95(pred_i, gt_i)

        # compute AJI
        C += overlap_area
        U += np.sum(pred_i) + np.sum(gt_i) - overlap_area

        # pred_copy[pred_i > 0] = 0
        pred_labeled[pred_i > 0] = 0  # remove the used nucleus

    # compute recall, precision, F1
    FP = Ns - TP
    recall = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    F1 = 2 * TP / (2 * TP + FP + FN + 1e-10)

    dice /= (count + 1e-8)
    iou /= (count + 1e-8)
    haus /= (count + 1e-8)
    hd_95 /= (count + 1e-8)

    # compute AJI
    U += np.sum(pred_labeled > 0)
    AJI = float(C) / U

    return recall, precision, F1, dice, iou, haus, AJI, hd_95


def gland_accuracy_object_level(pred, gt):
    """ Compute the object-level hausdorff distance between predicted  and
    groundtruth """

    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)

    # get connected components
    pred_labeled = morph.label(pred, connectivity=2)
    Ns = len(np.unique(pred_labeled)) - 1
    gt_labeled = morph.label(gt, connectivity=2)
    gt_labeled = morph.remove_small_objects(gt_labeled, 3)  # remove 1 or 2 pixel noise in the image
    gt_labeled = morph.label(gt_labeled, connectivity=2)
    Ng = len(np.unique(gt_labeled)) - 1

    # show_figures((pred_labeled, gt_labeled))

    # --- compute F1 --- #
    TP = 0.0  # true positive
    FP = 0.0  # false positive
    for i in range(1, Ns + 1):
        pred_i = np.where(pred_labeled == i, 1, 0)
        img_and = np.logical_and(gt_labeled, pred_i)

        # get intersection objects in target
        overlap_parts = img_and * gt_labeled
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        # show_figures((img_i, overlap_parts))

        # no intersection object
        if obj_no.size == 0:
            FP += 1
            continue

        # find max overlap object
        obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
        gt_obj = obj_no[np.argmax(obj_areas)]  # ground truth object number

        gt_obj_area = np.sum(gt_labeled == gt_obj)  # ground truth object area
        overlap_area = np.sum(overlap_parts == gt_obj)

        if float(overlap_area) / gt_obj_area >= 0.5:
            TP += 1
        else:
            FP += 1

    FN = Ng - TP  # false negative

    if TP == 0:
        precision = 0
        recall = 0
        F1 = 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)

    # --- compute dice, iou, hausdorff --- #
    pred_objs_area = np.sum(pred_labeled > 0)  # total area of objects in image
    gt_objs_area = np.sum(gt_labeled > 0)  # total area of objects in groundtruth gt

    # compute how well groundtruth object overlaps its segmented object
    dice_g = 0.0
    iou_g = 0.0
    hausdorff_g = 0.0
    hd_95_g = 0.0
    for i in range(1, Ng + 1):
        gt_i = np.where(gt_labeled == i, 1, 0)
        overlap_parts = gt_i * pred_labeled

        # get intersection objects numbers in image
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        gamma_i = float(np.sum(gt_i)) / gt_objs_area

        # show_figures((pred_labeled, gt_i, overlap_parts))

        if obj_no.size == 0:  # no intersection object
            dice_i = 0
            iou_i = 0

            # find nearest segmented object in hausdorff distance
            min_haus, min_hd95 = 1e5, 1e5
            for j in range(1, Ns + 1):
                pred_j = np.where(pred_labeled == j, 1, 0)
                seg_ind = np.argwhere(pred_j)
                gt_ind = np.argwhere(gt_i)
                haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])
                hd_95_tmp = hd95(pred_j, gt_i)
                if haus_tmp < min_haus:
                    min_haus = haus_tmp
                    min_hd95 = hd_95_tmp
            haus_i = min_haus
            haus95_i = min_hd95
        else:
            # find max overlap object
            obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
            seg_obj = obj_no[np.argmax(obj_areas)]  # segmented object number
            pred_i = np.where(pred_labeled == seg_obj, 1, 0)  # segmented object

            overlap_area = np.max(obj_areas)  # overlap area

            dice_i = 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
            iou_i = float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)

            # compute hausdorff distance
            seg_ind = np.argwhere(pred_i)
            gt_ind = np.argwhere(gt_i)
            haus_i = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])
            haus95_i = hd95(pred_i, gt_i)

        dice_g += gamma_i * dice_i
        iou_g += gamma_i * iou_i
        hausdorff_g += gamma_i * haus_i
        hd_95_g += gamma_i * haus95_i

    # compute how well segmented object overlaps its groundtruth object
    dice_s = 0.0
    iou_s = 0.0
    hausdorff_s = 0.0
    hd_95_s = 0.0
    for j in range(1, Ns + 1):
        pred_j = np.where(pred_labeled == j, 1, 0)
        overlap_parts = pred_j * gt_labeled

        # get intersection objects number in gt
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        # show_figures((pred_j, gt_labeled, overlap_parts))

        sigma_j = float(np.sum(pred_j)) / pred_objs_area
        # no intersection object
        if obj_no.size == 0:
            dice_j = 0
            iou_j = 0

            # find nearest groundtruth object in hausdorff distance
            min_haus, min_hd95 = 1e5, 1e5
            for i in range(1, Ng + 1):
                gt_i = np.where(gt_labeled == i, 1, 0)
                seg_ind = np.argwhere(pred_j)
                gt_ind = np.argwhere(gt_i)
                haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])
                hd_95_tmp = hd95(pred_j, gt_i)
                if haus_tmp < min_haus:
                    min_haus = haus_tmp
                    min_hd95 = hd_95_tmp
            haus_j = min_haus
            haus95_j = min_hd95
        else:
            # find max overlap gt
            gt_areas = [np.sum(overlap_parts == k) for k in obj_no]
            gt_obj = obj_no[np.argmax(gt_areas)]  # groundtruth object number
            gt_j = np.where(gt_labeled == gt_obj, 1, 0)  # groundtruth object

            overlap_area = np.max(gt_areas)  # overlap area

            dice_j = 2 * float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j))
            iou_j = float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j) - overlap_area)

            # compute hausdorff distance
            seg_ind = np.argwhere(pred_j)
            gt_ind = np.argwhere(gt_j)
            haus_j = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])
            haus95_j = hd95(pred_j, gt_j)

        dice_s += sigma_j * dice_j
        iou_s += sigma_j * iou_j
        hausdorff_s += sigma_j * haus_j
        hd_95_s += sigma_j * haus95_j

    return recall, precision, F1, (dice_g + dice_s) / 2, (iou_g + iou_s) / 2, (hausdorff_g + hausdorff_s) / 2, (
            hd_95_g + hd_95_s) / 2
