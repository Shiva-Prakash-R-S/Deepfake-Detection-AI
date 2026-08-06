import torch
import torch.nn as nn


class FFTBranch(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):

        # FFT Transform
        fft = torch.fft.fft2(x)

        # Magnitude Spectrum
        fft = torch.abs(fft)

        features = self.cnn(fft)

        return features.view(features.size(0), -1)


# Test
if __name__ == "__main__":

    model = FFTBranch()

    dummy = torch.randn(1, 3, 224, 224)

    output = model(dummy)

    print("=" * 50)
    print("FFT Branch Working Successfully!")
    print("=" * 50)
    print("Output Shape:", output.shape)