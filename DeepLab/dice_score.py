import torch
from torch import Tensor
from sklearn.metrics import roc_auc_score
import numpy as np
import torch.nn.functional as F

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def multiclass_dice_coeff(input: Tensor, target: Tensor, epsilon: float = 1e-7):
    #Compute the average Dice coefficient over all classes in a multi-class segmentation.
    #The epsilon represents a constant to prevent the division by zero
    
    batch_size, num_classes, H, W = input.shape
    dice_scores = []
    #Ensure input and target tensors are in the same device
    input = input.to(target.device)
    for class_idx in range(num_classes):
        #Create binary masks for the current class
        input_binary = (input[:, class_idx] > 0.5).float()
        target_binary = target[:, class_idx].float()
        #Calculate intersection and union for the current class
        intersection = (input_binary * target_binary).sum()
        union = input_binary.sum() + target_binary.sum()
        dice_score = (2. * intersection + epsilon) / (union + epsilon)
        dice_scores.append(dice_score)
    return torch.stack(dice_scores).mean().item()

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target)

def accuracy(output: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False):
    with torch.no_grad():
        if output.dim() == 4:
            output = output.argmax(dim=1)
        else:
            output = (output > 0.5).float()
        correct = (output == target).float()
        if reduce_batch_first:
            accuracy_per_batch = correct.view(correct.size(0), -1).mean(dim=1)
            return accuracy_per_batch.mean().item()
        else:
            return correct.mean().item()


def precision(output: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False):
    with torch.no_grad():
        if output.dim() == 4: 
            output = output.argmax(dim=1)
        else:
            output = (output > 0.5).float()
        #Initialize lists to store true positives and predicted positives for each class
        true_positives_per_class = []
        predicted_positives_per_class = []
        #Loop through each class
        for class_label in range(target.shape[1]):
            #Calculate true positives for the current class
            true_positives = ((output == class_label) & (target[:, class_label]==1)).sum()
            #Calculate predicted positives for the current class
            predicted_positives = (output == class_label).sum()
            true_positives_per_class.append(true_positives.item())
            predicted_positives_per_class.append(predicted_positives.item())
        true_positives_per_class = torch.tensor(true_positives_per_class, dtype=torch.float32)
        predicted_positives_per_class = torch.tensor(predicted_positives_per_class, dtype=torch.float32)
        precision_per_class = true_positives_per_class / (predicted_positives_per_class + 1e-7)  # Add epsilon to avoid division by zero
        if reduce_batch_first:
            return precision_per_class.mean().item()
        else:
            return precision_per_class.mean(dim=0).tolist()


def sensitivity(output: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False):
    with torch.no_grad():
        if output.dim() == 4:
            output = output.argmax(dim=1)
        else:
            output = (output > 0.5).float()
        true_positives_per_class = []
        false_negatives_per_class = []
        for class_label in range(target.shape[1]):
            true_positives = ((output == class_label) & (target[:,class_label]==1)).float().sum()
            false_negatives = ((output != class_label) & (target[:, class_label]==1)).float().sum()
            true_positives_per_class.append(true_positives.item())
            false_negatives_per_class.append(false_negatives.item())
        true_positives_per_class = torch.tensor(true_positives_per_class, dtype=torch.float32)
        false_negatives_per_class = torch.tensor(false_negatives_per_class, dtype=torch.float32)
        #Calculate sensitivity for each class
        sensitivity_per_class = true_positives_per_class / (true_positives_per_class + false_negatives_per_class + 1e-7)  # Add epsilon to avoid division by zero
        if reduce_batch_first:
            return sensitivity_per_class.mean().item()
        else:
            return sensitivity_per_class.mean(dim=0).tolist()

def specificity(output: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False):
    with torch.no_grad():
        if output.dim() == 4:
            output = output.argmax(dim=1)
        else:
            output = (output > 0.5).float()
        true_negatives_per_class = []
        false_positives_per_class = []
        for class_label in range(target.shape[1]):
            true_negatives = ((output != class_label) & (target[:, class_label] == 0)).float().sum()
            false_positives = ((output == class_label) & (target[:, class_label] == 0)).float().sum()
            true_negatives_per_class.append(true_negatives.item())
            false_positives_per_class.append(false_positives.item())
        true_negatives_per_class = torch.tensor(true_negatives_per_class, dtype=torch.float32)
        false_positives_per_class = torch.tensor(false_positives_per_class, dtype=torch.float32)
        #Calculate specificity for each class
        specificity_per_class = true_negatives_per_class / (true_negatives_per_class + false_positives_per_class + 1e-7)  # Add epsilon to avoid division by zero
        if reduce_batch_first:
            return specificity_per_class.mean().item()
        else:
            return specificity_per_class.mean(dim=0).tolist()


def false_positive_rate(output: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False):
    with torch.no_grad():
        if output.dim() == 4:
            output = output.argmax(dim=1)
        else:
            output = (output > 0.5).float()
        false_positives_per_class = []
        true_negatives_per_class = []
        for class_label in range(target.shape[1]):
            false_positives = ((output == class_label) & (target[:, class_label] == 0)).float().sum()
            true_negatives = ((output != class_label) & (target[:, class_label] == 0)).float().sum()
            false_positives_per_class.append(false_positives.item())
            true_negatives_per_class.append(true_negatives.item())
        false_positives_per_class = torch.tensor(false_positives_per_class, dtype=torch.float32)
        true_negatives_per_class = torch.tensor(true_negatives_per_class, dtype=torch.float32)
        #Calculate false positive rate for each class
        false_positive_rate_per_class = false_positives_per_class / (false_positives_per_class + true_negatives_per_class + 1e-7)  # Add epsilon to avoid division by zero
        if reduce_batch_first:
            return false_positive_rate_per_class.mean().item()
        else:
            return false_positive_rate_per_class.mean(dim=0).tolist()

def false_negative_rate(output: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False):
    with torch.no_grad():
        if output.dim() == 4:
            output = output.argmax(dim=1)
        else:
            output = (output > 0.5).float()
        false_negatives_per_class = []
        true_positives_per_class = []
        for class_label in range(target.shape[1]):
            false_negatives = ((output != class_label) & (target[:, class_label] == 1)).float().sum()
            true_positives = ((output == class_label) & (target[:, class_label] == 1)).float().sum()
            false_negatives_per_class.append(false_negatives.item())
            true_positives_per_class.append(true_positives.item())
        false_negatives_per_class = torch.tensor(false_negatives_per_class, dtype=torch.float32)
        true_positives_per_class = torch.tensor(true_positives_per_class, dtype=torch.float32)
        #Calculate false negative rate for each class
        false_negative_rate_per_class = false_negatives_per_class / (false_negatives_per_class + true_positives_per_class + 1e-7)  # Add epsilon to avoid division by zero
        if reduce_batch_first:
            return false_negative_rate_per_class.mean().item()
        else:
            return false_negative_rate_per_class.mean(dim=0).tolist()





