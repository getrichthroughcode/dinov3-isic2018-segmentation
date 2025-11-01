import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoV2Encoder(nn.Module):
    def __init__(self, model_name="dinov2_vits14", n_layers=12):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.embed_dim = self.backbone.embed_dim
        self.n_layers = n_layers

    def forward(self, x):
        feats = self.backbone.get_intermediate_layers(x, n=self.n_layers)
        idxs = [2, 5, 8, 11]
        outputs = []
        for i in idxs:
            feat = feats[i]
            B, N, C = feat.shape
            h = w = int(N**0.5)
            feat = feat.permute(0, 2, 1).contiguous().view(B, C, h, w)
            outputs.append(feat)
        return outputs  # [z3, z6, z9, z12]


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DinoUNet(nn.Module):
    def __init__(self, n_classes=1, encoder_name="dinov2_vits14"):
        super().__init__()
        self.encoder = DinoV2Encoder(encoder_name)
        C = self.encoder.embed_dim

        self.up1 = UpBlock(C, C, 512)
        self.up2 = UpBlock(512, C, 256)
        self.up3 = UpBlock(256, C, 128)
        self.up4 = UpBlock(128, C, 64)

        self.final_conv = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        z3, z6, z9, z12 = self.encoder(x)
        x = self.up1(z12, z9)
        x = self.up2(x, z6)
        x = self.up3(x, z3)
        x = self.up4(x, z3)
        x = self.final_conv(x)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x
