# cardiac-segmentation-xai

Multi-class whole heart segmentation on the MM-WHS dataset (CT and MRI), with explainability and interpretability analysis.

## What this project does

Trains and evaluates three segmentation models on 2D axial slices from the MM-WHS dataset:

- **Baseline**: ResNet-34 encoder, random initialisation, no attention, CE loss
- **ResU-Net**: ResNet-34 encoder, ImageNet pretrained, scSE attention, combined CE + Dice loss
- **EfficientNet-UNet**: EfficientNet-B4 encoder, same decoder as ResU-Net

Seven cardiac structures are segmented: LV, RV, LA, RA, MYO, AA, PA plus background (8 classes total).

After training, trustworthiness analysis notebooks run Grad-CAM++, scSE attention map visualisation, calibration with temperature scaling, and prediction entropy uncertainty maps.

## Dataset

MM-WHS (Multi-Modality Whole Heart Segmentation) dataset. Pre-processed 256x256 axial slices in `.npz` format with `image` and `label` arrays. Not included in this repo.

Expected structure:

```
pack/processed/
    ct_256/
        train/npz/
        val/npz/
        test/npz/
    mr_256/
        train/npz/
        val/npz/
        test/npz/
```

## Project structure

```
src/
    dataset.py      # MMWHSDataset, CLAHE preprocessing, augmentation
    model.py        # build_model() using segmentation_models_pytorch
    losses.py       # combined CE + Dice loss, class weight computation
    metrics.py      # Dice, Jaccard, ASD, HD
config/
    setup_env.py    # environment setup
Notebooks/
    segmentation_baseline_ct.ipynb
    segmentation_baseline_mr.ipynb
    segmentation_resunet_combined_loss_ct.ipynb
    segmentation_resunet_combined_loss_mr.ipynb
    segmentation_efficientnet_ct.ipynb
    segmentation_efficientnet_mr.ipynb
    trustworthy_analysis_ct.ipynb
    trustworthy_analysis_mr.ipynb
    visualise_data.ipynb
checkpoints/        # saved model weights (.pth)
```

## Dependencies

```
pip install -r requirements.txt
```

## Running

Open any training notebook and run cells in order. The `config/setup_env.py` script sets `PROJECT_ROOT` and `DATA_DIR` automatically. Checkpoints are saved to `checkpoints/`.

For trustworthy analysis, run `trustworthy_analysis_ct.ipynb` or `trustworthy_analysis_mr.ipynb` after training. Update `CKPT_PATH` in the config cell to point to your checkpoint.

## Evaluation

Models are evaluated on Dice, Jaccard, ASD, and HD per class and averaged across all eight classes including background. ASD and HD are reported in pixels.
