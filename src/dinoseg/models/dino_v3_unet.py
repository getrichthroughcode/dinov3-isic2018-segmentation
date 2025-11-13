import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DinoV3Encoder(nn.Module):
    def __init__(self, model_name="dinov3_vits16", weights_path=None, n_layers=12):
        super().__init__()

        model_name_map = {
            "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
            "dinov3_vitb16": "dinov3_base_patch16_518",
            "dinov3_vitl16": "dinov3_large_patch16_518",
        }
        timm_model_name = model_name_map.get(model_name, model_name)
        str(timm_model_name)
        out_indices = (2, 5, 8, 11)

        self.backbone = timm.create_model(
            "vit_small_patch16_dinov3.lvd1689m",
            pretrained=True,
            features_only=True,
            out_indices=out_indices,
        )

        self.embed_dim = self.backbone.model.embed_dim

    def forward(self, x):
        feats = self.backbone(x)
        return feats


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
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


class Dinov3UNet(nn.Module):
    def __init__(self, n_classes=1, encoder_name="dinov3_vits16"):
        super().__init__()
        self.encoder = DinoV3Encoder(
            encoder_name,
        )
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
