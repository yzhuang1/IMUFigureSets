from torch import nn

class SmallCNN(nn.Module):
    def __init__(self, in_channels: int=1, num_classes: int=2, hidden: int=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*8*8, hidden), nn.ReLU(),  # assumes >= 32x32 input
            nn.Linear(hidden, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x
