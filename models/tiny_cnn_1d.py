from torch import nn

class TinyCNN1D(nn.Module):
    # Expect input [B, T, C]. We'll transpose to [B, C, T] internally.
    def __init__(self, in_channels: int=6, num_classes: int=2, hidden: int=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )
        
    def forward(self, x, lengths=None):
        if x.dim()==3:
            x = x.permute(0, 2, 1)  # [B,T,C] -> [B,C,T]
        x = self.conv(x)
        x = self.head(x)
        return x
