import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from dice_score import multiclass_dice_coeff, dice_coeff, precision, sensitivity, specificity, false_positive_rate, false_negative_rate

@torch.inference_mode()
def evaluate_end_of_epoch(net, dataloader, device, amp, n_classes, criterion):
    net.eval()
    num_val_batches = len(dataloader)
    dice_scores = []
    accuracies = []
    precisions = []
    sensitivities = []
    specificities = []
    fprs = []
    fnrs = []
    dices = {class_idx: [] for class_idx in range(1, n_classes)}
    losses=[]
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            output = net(image)
            mask_pred = output['out']
            loss = criterion(mask_pred, mask_true)
            losses.append(loss.item())
            if n_classes == 1:
                mask_pred = (mask_pred > 0.5).float()
                dice_score = dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                dice_scores.append(dice_score.item())
                accuracy = (mask_pred == mask_true).float().mean()
                accuracies.append(accuracy.item())
                precision_score = precision(mask_pred, mask_true, reduce_batch_first=False)
                precisions.append(precision_score)
                sensitivity_score = sensitivity(mask_pred, mask_true, reduce_batch_first=False)
                sensitivities.append(sensitivity_score)
                specificity_score = specificity(mask_pred, mask_true, reduce_batch_first=False)
                specificities.append(specificity_score)
                fpr_score = false_positive_rate(mask_pred, mask_true, reduce_batch_first=False)
                fprs.append(fpr_score)
                fnr_score = false_negative_rate(mask_pred, mask_true, reduce_batch_first=False)
                fnrs.append(fnr_score)
                dices[1].append(dice_score.item())
            else:
                mask_true_one_hot = F.one_hot(mask_true, n_classes).permute(0, 3, 1, 2).float()
                mask_pred_one_hot = F.one_hot(mask_pred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
                dice_score = multiclass_dice_coeff(mask_pred_one_hot, mask_true_one_hot)
                dice_scores.append(dice_score)
                accuracy = (mask_pred.argmax(dim=1) == mask_true).float().mean()
                accuracies.append(accuracy.item())
                precision_score = precision(mask_pred_one_hot, mask_true_one_hot, reduce_batch_first=False)
                precisions.append(precision_score)
                sensitivity_score = sensitivity(mask_pred_one_hot, mask_true_one_hot, reduce_batch_first=False)
                sensitivities.append(sensitivity_score)
                specificity_score = specificity(mask_pred_one_hot, mask_true_one_hot, reduce_batch_first=False)
                specificities.append(specificity_score)
                fpr_score = false_positive_rate(mask_pred_one_hot, mask_true_one_hot, reduce_batch_first=False)
                fprs.append(fpr_score)
                fnr_score = false_negative_rate(mask_pred_one_hot, mask_true_one_hot, reduce_batch_first=False)
                fnrs.append(fnr_score)
                for class_idx in range(1, n_classes):
                    pred_mask_class = mask_pred_one_hot[:, class_idx]
                    true_mask_class = mask_true_one_hot[:, class_idx]
                    dice = dice_coeff(pred_mask_class, true_mask_class, reduce_batch_first=False)
                    dices[class_idx].append(dice.item())

    net.train()
    avg_dice_score = np.mean(dice_scores) if dice_scores else 0
    avg_accuracy = np.mean(accuracies) if accuracies else 0
    avg_precision = np.mean(precisions) if precisions else 0
    avg_sensitivity = np.mean(sensitivities) if sensitivities else 0
    avg_specificity = np.mean(specificities) if specificities else 0
    avg_fpr = np.mean(fprs) if fprs else 0
    avg_fnr = np.mean(fnrs) if fnrs else 0
    avg_dice_per_class = {class_idx: np.mean(dice_list) for class_idx, dice_list in dices.items()}
    avg_loss = np.mean(losses) if losses else 0

    return {
        'avg_dice_score': avg_dice_score,
        'avg_accuracy': avg_accuracy,
        'avg_precision': avg_precision,
        'avg_sensitivity': avg_sensitivity,
        'avg_specificity': avg_specificity,
        'avg_fpr': avg_fpr,
        'avg_fnr': avg_fnr,
        'avg_dice_per_class': avg_dice_per_class,
        'avg_loss': avg_loss,
    }

@torch.inference_mode()
def evaluate_per_image(net, image, mask_true, device, amp, n_classes):
    net.eval()
    dice_scores = []
    accuracies = []
    precisions = []
    sensitivities = []
    specificities = []
    fprs = []
    fnrs = []
    dices = {class_idx: [] for class_idx in range(1, n_classes)}

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        output = net(image)
        mask_pred = output['out']

        if n_classes == 1:
            mask_pred = (mask_pred > 0.5).float()
            dice_score = dice_coeff(mask_pred, mask_true)
            dice_scores.append(dice_score)
            accuracy = (mask_pred == mask_true).float().mean()
            accuracies.append(accuracy.item())
            precision_score = precision(mask_pred, mask_true, reduce_batch_first=False)
            precisions.append(precision_score)
            sensitivity_score = sensitivity(mask_pred, mask_true, reduce_batch_first=False)
            sensitivities.append(sensitivity_score)
            specificity_score = specificity(mask_pred, mask_true, reduce_batch_first=False)
            specificities.append(specificity_score)
            fpr_score = false_positive_rate(mask_pred, mask_true, reduce_batch_first=False)
            fprs.append(fpr_score)
            fnr_score = false_negative_rate(mask_pred, mask_true, reduce_batch_first=False)
            fnrs.append(fnr_score)
            dices[1].append(dice_score.item())
        else:
            mask_true_one_hot = F.one_hot(mask_true, n_classes).permute(0, 3, 1, 2).float()
            mask_pred_one_hot = F.one_hot(mask_pred.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
            dice_score = multiclass_dice_coeff(mask_pred_one_hot, mask_true_one_hot)
            dice_scores.append(dice_score)
            accuracy = (mask_pred.argmax(dim=1) == mask_true).float().mean()
            accuracies.append(accuracy.item())
            precision_score = precision(mask_pred_one_hot, mask_true_one_hot, reduce_batch_first=False)
            precisions.append(precision_score)
            sensitivity_score = sensitivity(mask_pred_one_hot, mask_true_one_hot, reduce_batch_first=False)
            sensitivities.append(sensitivity_score)
            specificity_score = specificity(mask_pred_one_hot, mask_true_one_hot, reduce_batch_first=False)
            specificities.append(specificity_score)
            fpr_score = false_positive_rate(mask_pred_one_hot, mask_true_one_hot, reduce_batch_first=False)
            fprs.append(fpr_score)
            fnr_score = false_negative_rate(mask_pred_one_hot, mask_true_one_hot, reduce_batch_first=False)
            fnrs.append(fnr_score)
            for class_idx in range(1, n_classes):
                pred_mask_class = mask_pred_one_hot[:, class_idx]
                true_mask_class = mask_true_one_hot[:, class_idx]
                dice = dice_coeff(pred_mask_class, true_mask_class, reduce_batch_first=False)
                dices[class_idx].append(dice.item())

    avg_dice_score = np.mean(dice_scores) if dice_scores else 0
    avg_accuracy = np.mean(accuracies) if accuracies else 0
    avg_precision = np.mean(precisions) if precisions else 0
    avg_sensitivity = np.mean(sensitivities) if sensitivities else 0
    avg_specificity = np.mean(specificities) if specificities else 0
    avg_fpr = np.mean(fprs) if fprs else 0
    avg_fnr = np.mean(fnrs) if fnrs else 0
    avg_dice_per_class = {class_idx: np.mean(dice_list) for class_idx, dice_list in dices.items()}

    return {
        'avg_dice_score': avg_dice_score,
        'avg_accuracy': avg_accuracy,
        'avg_precision': avg_precision,
        'avg_sensitivity': avg_sensitivity,
        'avg_specificity': avg_specificity,
        'avg_fpr': avg_fpr,
        'avg_fnr': avg_fnr,
        'avg_dice_per_class': avg_dice_per_class,
    }

