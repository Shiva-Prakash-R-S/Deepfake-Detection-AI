import torch
import torch.nn as nn

from .efficientnet_model import EfficientNetFeatureExtractor
from .fft_branch import FFTBranch
from .attention_fusion import AttentionFusion

class DeepfakeDetector(nn.Module):

    def __init__(self):
        super().__init__()

        self.efficientnet = EfficientNetFeatureExtractor()
        self.fft = FFTBranch()
        self.fusion = AttentionFusion()

    def forward(self, x):

        spatial_features = self.efficientnet(x)

        frequency_features = self.fft(x)

        output = self.fusion(
            spatial_features,
            frequency_features
        )

        return output


# Test
if __name__ == "__main__":

    model = DeepfakeDetector()

    dummy = torch.randn(2, 3, 224, 224)

    output = model(dummy)

    print("=" * 50)
    print("Complete Deepfake Model Working!")
    print("=" * 50)
    print("Output Shape:", output.shape)