import warnings
import numpy as np
import cv2
import skimage.measure as measure
import skimage.morphology as morphology
from scipy.ndimage.morphology import binary_fill_holes
warnings.filterwarnings("ignore")

# Dice Score & IoU for binary segmentation
class Evaluator(object):
    def __init__(self):
        self.Dice = 0
        self.IoU = 0
        self.num_batch = 0
        self.eps = 1e-4

        self.cell_num = 0
        self.iou_per_img = list()
        self.dice_per_cell = list()
        self.AJI = 0

    def dice_fn(self, gt_image, pre_image):
        eps = 1e-4
        batch_size = pre_image.shape[0]

        pre_image = pre_image.reshape(batch_size, -1).astype(np.bool)
        gt_image = gt_image.reshape(batch_size, -1).astype(np.bool)

        intersection = np.logical_and(pre_image, gt_image).sum(axis=1)
        union = pre_image.sum(axis=1) + gt_image.sum(axis=1) + eps
        Dice = ((2. * intersection + eps) / union).mean()
        IoU = Dice / (2. - Dice)
        return Dice, IoU

    # https://github.com/naivete5656/WSISPDR/blob/master/utils/for_review.py
    def mdice_fn(self, target, pred):
        '''
        :param target: hxw label
        :param pred: hxw label
        :return: mIoU, mDice
        '''
        iou_mean = 0.
        dice_mean = 0.
        self.cell_num += len(np.unique(target)[1:])
        # for idx, target_label in enumerate(range(1, target.max() + 1)):
        for idx, target_label in enumerate(np.unique(target)[1:]):
            if np.sum(target == target_label) < 20:
                target[target == target_label] = 0
                # seek pred label correspond to the label of target
            correspond_labels = pred[target == target_label]
            correspond_labels = correspond_labels[correspond_labels != 0]
            unique, counts = np.unique(correspond_labels, return_counts=True)
            try:
                max_label = unique[counts.argmax()]
                pred_mask = np.zeros(pred.shape)
                pred_mask[pred == max_label] = 1
            except ValueError:
                bou_list = []
                max_bou = target.shape[0]
                max_bou_h = target.shape[1]
                bou_list.extend(target[0, :])
                bou_list.extend(target[max_bou - 1, :])
                bou_list.extend(target[:, max_bou_h - 1])
                bou_list.extend(target[:, 0])
                bou_list = np.unique(bou_list)
                for x in bou_list:
                    target[target == x] = 0
                pred_mask = np.zeros(pred.shape)

            # create mask
            target_mask = np.zeros(target.shape)
            target_mask[target == target_label] = 1
            pred_mask = pred_mask.flatten()
            target_mask = target_mask.flatten()

            tp = pred_mask.dot(target_mask)
            fn = pred_mask.sum() - tp
            fp = target_mask.sum() - tp
            
            precision = (tp + self.eps) / (tp + fp + self.eps)
            recall = (tp + self.eps) / (tp + fn + + self.eps)

            iou = ((tp + self.eps) / (tp + fp + fn + self.eps))
            dice = (2 * tp + self.eps) / (2 * tp + fn + fp + self.eps)
            iou_mean = (iou_mean * idx + iou) / (idx + 1)
            dice_mean = (dice_mean * idx + dice) / (idx + 1)
            
            self.dice_per_cell.append(dice)
            # self.iou_per_img.append(iou)

        return dice_mean, iou_mean

    def add_pred(self, gt_image, pre_image):
        pre_image = measure.label(pre_image)
        pre_image = morphology.remove_small_objects(pre_image, min_size=64)
        pre_image = binary_fill_holes(pre_image > 0)
        pre_image = measure.label(pre_image)

        batch_mdice, batch_miou = self.mdice_fn(gt_image, pre_image)
        # batch_aji = self.get_fast_aji(gt_image, pre_image)
        self.iou_per_img.append(batch_miou)
        
        self.Dice = (self.Dice * self.num_batch + batch_mdice) / (self.num_batch + 1)
        self.IoU = (self.IoU * self.num_batch + batch_miou) / (self.num_batch + 1)
        # self.AJI = (self.AJI * self.num_batch + batch_aji) / (self.num_batch + 1)
        self.num_batch += 1

    def SEM(self):
        # Squared Error of Mean
        iou_percentage = np.array(self.iou_per_img) * 100
        SEM = np.std(iou_percentage) / np.sqrt(self.cell_num)
        return SEM
    
    def reset(self):
        self.Dice = 0
        self.IoU = 0
        self.num_batch = 0

    def get_fast_aji(self, true, pred):
        """AJI version distributed by MoNuSeg, has no permutation problem but suffered from 
        over-penalisation similar to DICE2.
        Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
        not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no 
        effect on the result.
        """
        true = np.copy(true)  # ? do we need this
        pred = np.copy(pred)        
        
        true_id_list = list(np.unique(true))
        pred_id_list = list(np.unique(pred))

        for i, true_id in enumerate(true_id_list):
            if i!=true_id:
                # print(f'changing from {true_id} to {i}')
                true[true==true_id] = i
                true_id_list[i] = i 
        for j, pred_id in enumerate(pred_id_list):
            if j!=pred_id:
                # print(f'changing from {pred_id} to {j}')
                pred[pred==pred_id] = j
                pred_id_list[j] = j 

        true_masks = [
            None,
        ]
        for t in true_id_list[1:]:
            t_mask = np.array(true == t, np.uint8)
            true_masks.append(t_mask)
        pred_masks = [
            None,
        ]
        for p in pred_id_list[1:]:
            p_mask = np.array(pred == p, np.uint8)
            pred_masks.append(p_mask)

        # prefill with value
        pairwise_inter = np.zeros(
            [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
        )
        pairwise_union = np.zeros(
            [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
        )
        
        # caching pairwise
        for true_id in true_id_list[1:]:  # 0-th is background
            t_mask = true_masks[true_id]
            pred_true_overlap = pred[t_mask > 0]
            pred_true_overlap_id = np.unique(pred_true_overlap)
            pred_true_overlap_id = list(pred_true_overlap_id)
            for pred_id in pred_true_overlap_id:
                if pred_id == 0:  # ignore
                    continue  # overlaping background
                p_mask = pred_masks[pred_id]
                total = (t_mask + p_mask).sum()
                inter = (t_mask * p_mask).sum()
                pairwise_inter[true_id - 1, pred_id - 1] = inter
                pairwise_union[true_id - 1, pred_id - 1] = total - inter

        pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
        # pair of pred that give highest iou for each true, dont care
        # about reusing pred instance multiple times
        paired_pred = np.argmax(pairwise_iou, axis=1)
        pairwise_iou = np.max(pairwise_iou, axis=1)
        # exlude those dont have intersection
        paired_true = np.nonzero(pairwise_iou > 0.0)[0]
        paired_pred = paired_pred[paired_true]
        # print(paired_true.shape, paired_pred.shape)
        overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
        overall_union = (pairwise_union[paired_true, paired_pred]).sum()

        paired_true = list(paired_true + 1)  # index to instance ID
        paired_pred = list(paired_pred + 1)
        # add all unpaired GT and Prediction into the union
        unpaired_true = np.array(
            [idx for idx in true_id_list[1:] if idx not in paired_true]
        )
        unpaired_pred = np.array(
            [idx for idx in pred_id_list[1:] if idx not in paired_pred]
        )
        for true_id in unpaired_true:
            overall_union += true_masks[true_id].sum()
        for pred_id in unpaired_pred:
            overall_union += pred_masks[pred_id].sum()

        aji_score = overall_inter / overall_union
        return aji_score