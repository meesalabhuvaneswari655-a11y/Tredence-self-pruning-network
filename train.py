import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import PrunableMLP, PrunableLinear


def compute_sparsity_loss(model):
    loss = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            loss += gates.sum()
    return loss


def train(lambda_value=0.001, epochs=5):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )

    model = PrunableMLP()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        for images, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(images)

            classification_loss = criterion(outputs, labels)

            sparsity_loss = compute_sparsity_loss(model)

            total_loss = classification_loss + lambda_value * sparsity_loss

            total_loss.backward()

            optimizer.step()

            running_loss += total_loss.item()

        print(f"Epoch {epoch+1}: {running_loss:.2f}")

    torch.save(model.state_dict(), "prunable_model.pth")

    print("Training completed!")


if __name__ == "__main__":
    train(lambda_value=0.001, epochs=5)
