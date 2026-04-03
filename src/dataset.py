import os, glob, random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter, map_coordinates


def apply_clahe(image, clip_limit=0.3):
    """Normalise a float32 image to [0,255] uint8, apply CLAHE, return float32 [0,1]."""
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        img_u8 = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_u8 = np.zeros_like(image, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit * 255, tileGridSize=(8, 8))
    return clahe.apply(img_u8).astype(np.float32) / 255.0


def elastic_deform(img, lbl, alpha=30, sigma=4):
    """Smooth random displacement field applied identically to image and label."""
    h, w = img.shape
    dx = gaussian_filter(np.random.randn(h, w), sigma) * alpha
    dy = gaussian_filter(np.random.randn(h, w), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    coords = [np.clip(y + dy, 0, h - 1), np.clip(x + dx, 0, w - 1)]
    img_d = map_coordinates(img, coords, order=1, mode='reflect').astype(np.float32)
    lbl_d = map_coordinates(lbl.astype(np.float32), coords, order=0, mode='reflect').astype(np.uint8)
    return img_d, lbl_d


def augment_pair(img, lbl):
    """Random rotation ±20°, flips, intensity jitter, elastic deformation."""
    h, w = img.shape
    if random.random() > 0.5:
        angle = random.uniform(-20, 20)
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img   = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
        lbl   = cv2.warpAffine(lbl, M, (w, h), flags=cv2.INTER_NEAREST)
    if random.random() > 0.5:
        img, lbl = cv2.flip(img, 1), cv2.flip(lbl, 1)
    if random.random() > 0.5:
        img, lbl = cv2.flip(img, 0), cv2.flip(lbl, 0)
    if random.random() > 0.8:
        img, lbl = elastic_deform(img, lbl)
    if random.random() > 0.5:
        img = np.clip(img * random.uniform(0.8, 1.2), 0, 1).astype(np.float32)
    return img, lbl


class MMWHSDataset(Dataset):
    def __init__(self, modality, split, data_dir, augment=False, max_files=None, clahe_clip=0.3):
        folder     = os.path.join(data_dir, modality, split, 'npz')
        self.files = sorted(glob.glob(os.path.join(folder, '*.npz')))
        if max_files:
            self.files = random.sample(self.files, min(max_files, len(self.files)))
        self.augment    = augment
        self.clahe_clip = clahe_clip

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        img  = apply_clahe(data['image'].astype(np.float32), clip_limit=self.clahe_clip)
        lbl  = data['label'].astype(np.uint8)
        if self.augment:
            img, lbl = augment_pair(img, lbl)
        return torch.tensor(img).unsqueeze(0), torch.tensor(lbl.astype(np.int64))
