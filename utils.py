import torch
import matplotlib.pyplot as plt
from model import PrunableLinear


def calculate_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.numel()
            pruned += (gates < threshold).sum().item()

    return 100 * pruned / total


def plot_gate_distribution(model, save_path="gate_histogram.png"):
    all_gates = []

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            all_gates.extend(gates.flatten().detach().cpu().numpy())

    plt.figure(figsize=(6, 4))
    plt.hist(all_gates, bins=50)
    plt.xlabel("Gate Value")
    plt.ylabel("Count")
    plt.title("Distribution of Gate Values")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
