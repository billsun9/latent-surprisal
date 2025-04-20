import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        return self.linear(x)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, expand_factor = 2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_in, expand_factor * dim_in),
            nn.ReLU(),
            nn.Linear(expand_factor * dim_in, expand_factor * dim_out),
            nn.ReLU(),
            nn.Linear(expand_factor * dim_out, dim_out)
        )

    def forward(self, x):
        return self.net(x)