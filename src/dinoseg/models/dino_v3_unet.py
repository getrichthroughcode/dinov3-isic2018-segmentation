import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class DinoV3Encoder(nn.Module):
    def __init__(self, model_name="dinov3_vits16", weights_path=None, n_layers=12):
        super().__init__()
        repo_dir = torch.hub.get_dir() + "/facebookresearch_dinov3_main"
        self.backbone = torch.hub.load(repo_dir, model_name, source="local")
        if weights_path is not None:
            weights_path = Path(weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights not found : {weights_path}")
            state_dict = torch.load(weights_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]

            missing, unexpected = self.backbone.load_state_dict(
                state_dict, strict=False
            )
            print(f"[DINOv3] loaded from {weights_path}")
            if missing:
                print(f" - Missing Keys : {len(missing)}")
            if unexpected:
                print(f" - Unexpected Keys : {len(unexpected)}")

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
        return outputs


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
            weights_path="dinoseg/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
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
