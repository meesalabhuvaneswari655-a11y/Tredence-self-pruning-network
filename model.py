import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # Normal trainable weights
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02
        )

        # Trainable bias
        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )

        # Learnable gate scores
        self.gate_scores = nn.Parameter(
            torch.zeros(out_features, in_features)
        )

    def forward(self, x):
        # Convert gate scores into values between 0 and 1
        gates = torch.sigmoid(self.gate_scores)

        # Multiply weights by gates
        pruned_weight = self.weight * gates

        # Linear transformation
        return F.linear(x, pruned_weight, self.bias)


class PrunableMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
