from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def count_params(model: nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}


class DinoV3Encoder(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov3_vits16",
        pretrained: bool = True,
        out_indices: Tuple[int, int, int, int] = (2, 5, 8, 11),
    ):
        super().__init__()
        name_map = {
            "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
            "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",
            "dinov3_vitl16": "dinov3_large_patch16_dinov3.lvd1689m",
        }
        timm_name = name_map.get(model_name, model_name)
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        self.embed_dim = getattr(self.backbone.model, "embed_dim")

    def forward(self, x):
        feats = self.backbone(x)
        return tuple(feats[:])


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ks: int = 3, pad: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ks, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DINOAdapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if len(in_channels) != len(out_channels):
            raise ValueError("in_channels and out_channels must have same length.")
        self.projs = nn.ModuleList(
            [nn.Conv2d(i, o, kernel_size=1) for i, o in zip(in_channels, out_channels)]
        )
        self.refiners = nn.ModuleList([ConvBNReLU(o, o) for o in out_channels])

    def forward(self, feats):
        outs = []
        for f, p, r in zip(feats, self.projs, self.refiners):
            t = p(f)
            t = r(t)
            outs.append(t)
        return outs


class SharedContextAggregator(nn.Module):
    def __init__(self, in_channels, hidden):
        super().__init__()
        total = sum(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(total, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        pooled = [self.pool(f).flatten(1) for f in feats]
        concat = torch.cat(pooled, dim=1)
        return self.fc(concat)


class FAPM(nn.Module):
    def __init__(self, channels: int, context_dim: int):
        super().__init__()
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.ctx_proj = nn.Linear(context_dim, channels, bias=False)
        self.gate = nn.Sequential(nn.ReLU(inplace=True), nn.Sigmoid())

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        y = self.conv_spatial(x)
        b, c, h, w = y.shape
        ctx_v = self.ctx_proj(ctx).view(b, c, 1, 1)
        y = y + ctx_v
        g = self.gate(y.mean(dim=(2, 3)))
        g = g.view(b, c, 1, 1)
        return x + y * g


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv1 = ConvBNReLU(out_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Dinov3UNet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "dinov3_vits16",
        pretrained: bool = True,
        freeze_encoder: bool = True,
        adapter_out_channels: Tuple[int, int, int, int] = (256, 128, 64, 32),
        final_channels: int = 32,
        n_classes: int = 1,
    ):
        super().__init__()
        self.encoder = DinoV3Encoder(encoder_name, pretrained=pretrained)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        enc_ch = tuple([self.encoder.embed_dim] * 4)
        self.adapter = DINOAdapter(enc_ch, adapter_out_channels)

        shared_ctx_dim = 128
        self.ctx_agg = SharedContextAggregator(adapter_out_channels, shared_ctx_dim)
        self.fapms = nn.ModuleList(
            [FAPM(ch, shared_ctx_dim) for ch in adapter_out_channels]
        )

        p4, p3, p2, p1 = adapter_out_channels
        self.dec4 = nn.Sequential(ConvBNReLU(p4, p4))
        self.dec3 = DecoderBlock(in_ch=p4, skip_ch=p3, out_ch=p3)
        self.dec2 = DecoderBlock(in_ch=p3, skip_ch=p2, out_ch=p2)
        self.dec1 = DecoderBlock(in_ch=p2, skip_ch=p1, out_ch=p1)

        self.head_conv = nn.Sequential(
            ConvBNReLU(p1, final_channels),
            nn.Conv2d(final_channels, n_classes, kernel_size=1),
        )

    def forward(self, x):
        b, _, h, w = x.shape
        with torch.cuda.amp.autocast(enabled=False):
            x32 = x.float()
            feats = self.encoder(x32)
        adapts = self.adapter(feats)

        ctx = self.ctx_agg(adapts)
        fapm_outs = [m(a, ctx) for m, a in zip(self.fapms, adapts)]

        p4, p3, p2, p1 = fapm_outs

        x = self.dec4(p4)
        x = self.dec3(x, p3)
        x = self.dec2(x, p2)
        x = self.dec1(x, p1)

        out = self.head_conv(x)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out
