import torch
from model import PrunableMLP
from utils import calculate_sparsity, plot_gate_distribution

# Load trained model
model = PrunableMLP()
model.load_state_dict(torch.load("prunable_model.pth", map_location="cpu"))
model.eval()

# Calculate sparsity
sparsity = calculate_sparsity(model)
print(f"Sparsity: {sparsity:.2f}%")

# Save histogram
plot_gate_distribution(model, "results/histogram_best.png")
print("Histogram saved to results/histogram_best.png")
