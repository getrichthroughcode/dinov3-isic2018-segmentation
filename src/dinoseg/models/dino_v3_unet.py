import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}


class DinoV3Encoder(nn.Module):
    def __init__(
        self, model_name="dinov3_vits16", pretrained=True, out_indices=(2, 5, 8, 11)
    ):
        super().__init__()
        name_map = {
            "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
            "dinov3_vitb16": "dinov3_base_patch16_518",
            "dinov3_vitl16": "dinov3_large_patch16_518",
        }
        timm_name = name_map.get(model_name, model_name)
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        self.out_channels = self.backbone.feature_info.channels()
        self.embed_dim = int(self.out_channels[-1])

    def forward(self, x):
        return tuple(self.backbone(x)[:4])


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, pad=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ks, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, ks, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
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
    def __init__(
        self,
        n_classes=1,
        encoder_name="dinov3_vits16",
        pretrained=True,
        freeze_encoder=True,
        projector_channels=(32, 64, 128, 256),
    ):
        super().__init__()
        self.encoder = DinoV3Encoder(encoder_name, pretrained)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        p1_ch, p2_ch, p3_ch, p4_ch = projector_channels
        enc_ch = self.encoder.out_channels

        self.proj_p1 = nn.Conv2d(enc_ch[0], p1_ch, 1)
        self.proj_p2 = nn.Conv2d(enc_ch[1], p2_ch, 1)
        self.proj_p3 = nn.Conv2d(enc_ch[2], p3_ch, 1)
        self.proj_p4 = nn.Conv2d(enc_ch[3], p4_ch, 1)

        self.up1 = UpBlock(p4_ch, p3_ch, p3_ch)
        self.up2 = UpBlock(p3_ch, p2_ch, p2_ch)
        self.up3 = UpBlock(p2_ch, p1_ch, p1_ch)

        final_ch = max(p1_ch // 2, 16)
        self.up_final = nn.Sequential(
            nn.ConvTranspose2d(p1_ch, final_ch, 2, stride=2),
            nn.BatchNorm2d(final_ch),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(final_ch, n_classes, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        f1, f2, f3, f4 = self.encoder(x)

        p1 = self.proj_p1(f1)
        p2 = self.proj_p2(f2)
        p3 = self.proj_p3(f3)
        p4 = self.proj_p4(f4)

        h4, w4 = p4.shape[-2], p4.shape[-1]
        s1 = (h4, w4)
        s1 = s1
        s2 = (min(h4 * 2, h), min(w4 * 2, w))
        s3 = (min(h4 * 4, h), min(w4 * 4, w))
        s4 = (min(h4 * 8, h), min(w4 * 8, w))

        p3 = F.interpolate(p3, s2, mode="bilinear", align_corners=False)
        p2 = F.interpolate(p2, s3, mode="bilinear", align_corners=False)
        p1 = F.interpolate(p1, s4, mode="bilinear", align_corners=False)

        x = self.up1(p4, p3)
        x = self.up2(x, p2)
        x = self.up3(x, p1)

        x = self.up_final(x)
        x = self.final_conv(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x
