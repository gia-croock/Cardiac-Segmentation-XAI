import torch.nn as nn
import segmentation_models_pytorch as smp


def build_model(num_classes, device, dropout_p=0.15):
    """ResU-Net: smp U-Net with ResNet-34 encoder, scSE attention, segmentation-head dropout."""
    model = smp.Unet(
        encoder_name          = 'resnet34',
        encoder_weights       = 'imagenet',
        in_channels           = 1,
        classes               = num_classes,
        decoder_attention_type= 'scse',
        decoder_dropout       = 0.2,
    ).to(device)
    model.segmentation_head.add_module('dropout', nn.Dropout2d(p=dropout_p))
    return model
