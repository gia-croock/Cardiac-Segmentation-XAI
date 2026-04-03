import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_weights(data_dir, modality, num_classes, device):
    """Compute inverse-frequency class weights from all training labels."""
    counts = np.zeros(num_classes)
    for f in glob.glob(os.path.join(data_dir, modality, 'train/npz/*.npz')):
        lbl = np.load(f)['label']
        for c in range(num_classes):
            counts[c] += (lbl == c).sum()
    counts  = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32).to(device)


def dice_loss(preds, labels, num_classes, smooth=1e-6):
    """Soft Dice loss averaged over foreground classes."""
    probs = F.softmax(preds, dim=1)
    loss  = 0
    for c in range(1, num_classes):
        p = probs[:, c]
        t = (labels == c).float()
        loss += 1 - (2 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)
    return loss / (num_classes - 1)


def get_loss(name, weights, num_classes):
    """Return a loss function by name.

    Args:
        name: 'combined' | 'dice' | 'ce'
        weights: class weight tensor for CrossEntropyLoss
        num_classes: number of output classes
    """
    criterion = nn.CrossEntropyLoss(weight=weights)
    if name == 'combined':
        return lambda p, l: 0.5 * criterion(p, l) + 0.5 * dice_loss(p, l, num_classes)
    if name == 'dice':
        return lambda p, l: dice_loss(p, l, num_classes)
    if name == 'ce':
        return lambda p, l: criterion(p, l)
    raise ValueError(f"Unknown loss '{name}'. Choose from: 'combined', 'dice', 'ce'")
