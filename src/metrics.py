import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_score(pred, lbl, num_classes=8):
    """Mean foreground Dice across all classes present in the label."""
    scores = []
    for c in range(1, num_classes):
        g = (lbl == c)
        if g.sum() == 0:
            continue
        p = (pred == c)
        scores.append(2 * (p & g).sum() / (p.sum() + g.sum() + 1e-8))
    return float(np.mean(scores)) if scores else 0.0


def dice_binary(pred, gt):
    """Dice coefficient for a single binary class mask."""
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return (2 * inter) / denom if denom > 0 else float('nan')


def jaccard(pred, gt):
    """Jaccard (IoU) for a single binary class mask."""
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return inter / union if union > 0 else float('nan')


def surface_distances(pred, gt):
    """Average Surface Distance and Hausdorff Distance for a single binary class mask."""
    if pred.sum() == 0 or gt.sum() == 0:
        return float('nan'), float('nan')
    pred_border = pred ^ binary_erosion(pred)
    gt_border   = gt   ^ binary_erosion(gt)
    d1    = distance_transform_edt(~gt_border)[pred_border]
    d2    = distance_transform_edt(~pred_border)[gt_border]
    all_d = np.concatenate([d1, d2])
    return all_d.mean(), all_d.max()
