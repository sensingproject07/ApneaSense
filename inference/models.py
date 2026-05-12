"""
Model class definitions — must exactly match training-time architectures.

Audio  : MelCNN   (ResNet18, 1-channel input, binary output)
Vision : AttentionFusionClassifier (DepthEncoder + RGBEncoder + JointEncoder + Transformer)
"""
import torch
import torch.nn as nn
from torchvision import models as tv_models


# ── Audio model ────────────────────────────────────────────────────────────────

class MelCNN(nn.Module):
    """
    Binary apnea classifier.
    Input : (B, 1, 128, 313) log-Mel spectrogram normalised to [0, 1].
    Output: (B, 2) raw logits.
    """
    def __init__(self):
        super().__init__()
        self.input_norm = nn.BatchNorm2d(1)

        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        # Average pretrained 3-channel conv1 weights to 1 channel (matches training)
        old_w = backbone.conv1.weight.data          # (64, 3, 7, 7)
        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv1.weight.data = old_w.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv1

        self.encoder    = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 512, 1, 1)
        self.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(512, 2))

    def forward(self, x):
        x = self.input_norm(x)
        x = self.encoder(x).flatten(1)
        return self.classifier(x)


# ── Vision model ───────────────────────────────────────────────────────────────

class DepthEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = tv_models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(self.encoder(x))


class RGBEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = tv_models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone   = backbone
        self.projection = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 128),
        )

    def forward(self, x):
        return self.projection(self.backbone(x))


class JointEncoder(nn.Module):
    def __init__(self, input_dim=42, hidden_dim=128, feature_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.encoder(x)


class AttentionFusionClassifier(nn.Module):
    """
    Forward signature: model(depth_x, rgb_x, joint_x)  — depth FIRST.
    Input shapes: depth (B,1,224,224), rgb (B,3,224,224), joint (B,42).
    Output: (B, 3) raw logits  [supine, left, right].
    """
    def __init__(
        self,
        depth_encoder, rgb_encoder, joint_encoder,
        depth_feature_dim=512, rgb_feature_dim=128, joint_feature_dim=128,
        common_feature_dim=256, num_heads=4, ff_dim=512, dropout=0.1,
        num_classes=3,
    ):
        super().__init__()
        self.depth_encoder = depth_encoder
        self.rgb_encoder   = rgb_encoder
        self.joint_encoder = joint_encoder

        self.depth_proj = nn.Linear(depth_feature_dim,  common_feature_dim)
        self.rgb_proj   = nn.Linear(rgb_feature_dim,    common_feature_dim)
        self.joint_proj = nn.Linear(joint_feature_dim,  common_feature_dim)

        self.modality_embed = nn.Parameter(torch.randn(3, common_feature_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=common_feature_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.attention_block = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.classifier = nn.Sequential(
            nn.Linear(common_feature_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, depth_x, rgb_x, joint_x):
        with torch.no_grad():
            f_depth = self.depth_encoder(depth_x)
            f_rgb   = self.rgb_encoder(rgb_x)
            f_joint = self.joint_encoder(joint_x)
        t_depth = self.depth_proj(f_depth)
        t_rgb   = self.rgb_proj(f_rgb)
        t_joint = self.joint_proj(f_joint)
        tokens  = torch.stack([t_rgb, t_depth, t_joint], dim=1)
        tokens  = tokens + self.modality_embed.unsqueeze(0)
        tokens  = self.attention_block(tokens)
        return self.classifier(tokens.mean(dim=1))
