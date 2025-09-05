from torch import nn

class TabMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int=2, hidden: int=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )
        
    def forward(self, x):
        return self.net(x)
