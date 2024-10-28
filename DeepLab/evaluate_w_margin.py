import torch
import numpy as np
import cv2
import torch.nn.functional as F
import tqdm
from dice_score import multiclass_dice_coeff, dice_coeff, precision, sensitivity, specificity, false_positive_rate, false_negative_rate

#dilation_mm and erosion_mm must be modified according to the intended value to dilate or erode the ground-truth masks
#The pixels per mm in the data used for this purpose must be changed as well.
def dilate_mask(mask, pixels_per_mm=3.78, dilation_mm=0.5):
    kernel_size = int(pixels_per_mm * dilation_mm)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_np = mask.cpu().numpy().astype(np.uint8)
    dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)
    return torch.tensor(dilated_mask, dtype=mask.dtype, device=mask.device)

def erode_mask(mask, pixels_per_mm=3.78, erosion_mm=0.5):
    kernel_size = int(pixels_per_mm * erosion_mm)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_np = mask.cpu().numpy().astype(np.uint8)
    eroded_mask = cv2.erode(mask_np, kernel, iterations=1)
    return torch.tensor(eroded_mask, dtype=mask.dtype, device=mask.device)


@torch.inference_mode()
def evaluate_per_image(net, image, mask_true, n_classes, device, amp=False, apply_margin=True):
    net.eval()
    count_eroded_dice=0
    count_dilated_dice=0
    count_normal_dice=0
    dice_scores_classes = {class_idx: [] for class_idx in range(0, n_classes)}
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        output = net(image)
        mask_pred = output['out']

        dilated_mask = dilate_mask(mask_true)
        eroded_mask = erode_mask(mask_true)

        if net.n_classes == 1:
            mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
            initial_dice= dice_coeff(mask_pred, mask_true, reduce_batch_first=False)

            if apply_margin:
                eroded_dice = dice_coeff(mask_pred, eroded_mask, reduce_batch_first=False)
                dilated_dice = dice_coeff(mask_pred, dilated_mask, reduce_batch_first=False)
                if eroded_dice > initial_dice and eroded_dice > dilated_dice:
                    mask_true = eroded_mask
                elif dilated_dice > initial_dice:
                    mask_true = dilated_mask

            dice_score = dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            accuracy = (mask_pred == mask_true).float().mean()
            precision_score = precision(mask_pred, mask_true, reduce_batch_first=False)
            sensitivity_score = sensitivity(mask_pred, mask_true, reduce_batch_first=False)
            specificity_score = specificity(mask_pred, mask_true, reduce_batch_first=False)
            fpr_score = false_positive_rate(mask_pred, mask_true, reduce_batch_first=False)
            fnr_score = false_negative_rate(mask_pred, mask_true, reduce_batch_first=False)

        else:
            mask_true_one_hot = F.one_hot(mask_true, n_classes).permute(0, 3, 1, 2).float()
            mask_pred_one_hot = F.one_hot(mask_pred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()

            mask_true_class_0 = mask_true_one_hot[:, 0]  # Extracts class 0 mask (background)
            mask_true_class_1 = mask_true_one_hot[:, 1]  # Extracts class 1 mask (bone)
            mask_true_class_2 = mask_true_one_hot[:, 2]  # Extracts class 2 mask (ligament)

            mask_pred_class_0 = mask_pred_one_hot[:, 0]  # Extracts class 0 prediction (background)
            mask_pred_class_1 = mask_pred_one_hot[:, 1]  # Extracts class 1 prediction (bone)
            mask_pred_class_2 = mask_pred_one_hot[:, 2]  # Extracts class 2 prediction (ligament)

            dice_class_0 = dice_coeff(mask_pred_class_0, mask_true_class_0, reduce_batch_first=False)
            dice_class_1 = dice_coeff(mask_pred_class_1, mask_true_class_1, reduce_batch_first=False)
            dice_class_2 = dice_coeff(mask_pred_class_2, mask_true_class_2, reduce_batch_first=False)
            dice_scores_classes[0] = dice_class_0.item()
            dice_scores_classes[1] = dice_class_1.item()
            dice_scores_classes[2] = dice_class_2.item()

            initial_dice= multiclass_dice_coeff(mask_pred_one_hot, mask_true_one_hot)
            dice=initial_dice
            
            if apply_margin:
                eroded_mask = F.one_hot(eroded_mask, n_classes).permute(0, 3, 1, 2).float()
                eroded_dice = multiclass_dice_coeff(mask_pred_one_hot, eroded_mask)
                dilated_mask = F.one_hot(dilated_mask, n_classes).permute(0, 3, 1, 2).float()
                dilated_dice = multiclass_dice_coeff(mask_pred_one_hot, dilated_mask)
            
                if eroded_dice > initial_dice and eroded_dice > dilated_dice:
                    mask_true_one_hot = eroded_mask
                    dice=eroded_dice
                    dice_class_1 = dice_coeff(mask_pred_one_hot[:,1], mask_true_one_hot[:,1], reduce_batch_first=False)
                    dice_class_2 = dice_coeff(mask_pred_one_hot[:,2], mask_true_one_hot[:,2], reduce_batch_first=False)
                    count_eroded_dice= count_eroded_dice +1

                elif dilated_dice > initial_dice:
                    mask_true_one_hot = dilated_mask
                    dice=dilated_dice
                    dice_class_1 = dice_coeff(mask_pred_one_hot[:, 1], mask_true_one_hot[:,1], reduce_batch_first=False)
                    dice_class_2 = dice_coeff(mask_pred_one_hot[:,2], mask_true_one_hot[:,2], reduce_batch_first=False)
                    count_dilated_dice=count_dilated_dice+1
                else:
                    count_normal_dice=count_normal_dice+1

                dice_scores_classes[1] = dice_class_1.item()
                dice_scores_classes[2] = dice_class_2.item()

        dice_score = dice
        accuracy = (mask_pred_one_hot.argmax(dim=1) == mask_true).float().mean()
        precision_score = precision(mask_pred_one_hot, mask_true_one_hot, reduce_batch_first=False)
        sensitivity_score = sensitivity(mask_pred_one_hot, mask_true_one_hot, reduce_batch_first=False)
        specificity_score = specificity(mask_pred_one_hot, mask_true_one_hot, reduce_batch_first=False)
        fpr_score = false_positive_rate(mask_pred_one_hot, mask_true_one_hot, reduce_batch_first=False)
        fnr_score = false_negative_rate(mask_pred_one_hot, mask_true_one_hot, reduce_batch_first=False)

        results = {
            'dice_score': dice_score.item() if isinstance(dice_score, torch.Tensor) else dice_score,
            'accuracy': accuracy.item() if isinstance(accuracy, torch.Tensor) else accuracy,
            'precision': precision_score.item() if isinstance(precision_score, torch.Tensor) else precision_score,
            'sensitivity': sensitivity_score.item() if isinstance(sensitivity_score, torch.Tensor) else sensitivity_score,
            'specificity': specificity_score.item() if isinstance(specificity_score, torch.Tensor) else specificity_score,
            'fpr': fpr_score.item() if isinstance(fpr_score, torch.Tensor) else fpr_score,
            'fnr': fnr_score.item() if isinstance(fnr_score, torch.Tensor) else fnr_score,
            'count_eroded_dice': count_eroded_dice,
            'count_dilated_dice': count_dilated_dice,
            'count_normal_dice': count_normal_dice,
            'class_0_dice': dice_scores_classes[0],
            'class_1_dice': dice_scores_classes[1],
            'class_2_dice': dice_scores_classes[2],
        }
    return results
