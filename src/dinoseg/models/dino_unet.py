import torch.nn as nn
import timm


class DinoV3Encoder(nn.Module):
    def __init__(self, model_name="dinov3_vit16"):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=True, features_only=True
        )
        self.out_channels = self.backbone.feature_info[-1]["num_chs"]

    def forward(self, x):
        feats = self.backbone(x)
        return feats


class DinoUnet(nn.Module):
    def __init__(self, n_classes=1, encoder_name="dinov3_vit16"):
        super().__init__()
        self.encoder = DinoV3Encoder(encoder_name)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.encoder.out_channels, 256, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, n_classes),
        )

    def forward(self, x):
        feats = self.encoder(x)
        x = feats[-1]
        x = self.decoder(x)
        x = nn.functional.interpolate(
            x,
            size=(x.shape[-2] * 4, x.shape[-1] * 4),
            mode="bilinear",
            align_corners=False,
        )
        return x
