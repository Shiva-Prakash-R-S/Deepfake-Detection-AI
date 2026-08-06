import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()

        # EfficientNet = 1280
        # FFT = 128
        self.attention = nn.Sequential(
            nn.Linear(1408, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 1408),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1408, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 2)
        )

    def forward(self, efficientnet_features, fft_features):

        fused = torch.cat(
            [efficientnet_features, fft_features],
            dim=1
        )

        attention = self.attention(fused)

        fused = fused * attention

        output = self.classifier(fused)

        return output


# Test
if __name__ == "__main__":

    efficientnet = torch.randn(1, 1280)

    fft = torch.randn(1, 128)

    model = AttentionFusion()

    output = model(efficientnet, fft)

    print("=" * 50)
    print("Attention Fusion Working Successfully!")
    print("=" * 50)
    print("Output Shape:", output.shape)