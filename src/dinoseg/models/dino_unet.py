import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoV2Encoder(nn.Module):
    def __init__(self, model_name="dinov2_vits14"):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.embed_dim = self.backbone.embed_dim

    def forward(self, x):
        feats = self.backbone.get_intermediate_layers(x, n=1)
        last_feat = feats[-1]
        B, N, C = last_feat.shape
        h = w = int(N**0.5)
        last_feat = last_feat.permute(0, 2, 1).contiguous().view(B, C, h, w)
        return last_feat


class DinoUNet(nn.Module):
    def __init__(self, n_classes=1, encoder_name="dinov2_vits14"):
        super().__init__()
        self.encoder = DinoV2Encoder(encoder_name)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.encoder.embed_dim, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_classes, kernel_size=1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        feats = self.encoder(x)
        x = self.decoder(feats)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x
