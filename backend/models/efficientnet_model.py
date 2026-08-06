import torch
import torch.nn as nn
import timm


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained EfficientNet-B0
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0
        )

    def forward(self, x):
        features = self.backbone(x)
        return features


# Test the model
if __name__ == "__main__":

    model = EfficientNetFeatureExtractor()

    dummy = torch.randn(1, 3, 224, 224)

    output = model(dummy)

    print("=" * 50)
    print("EfficientNet-B0 Loaded Successfully!")
    print("=" * 50)
    print("Output Shape:", output.shape)