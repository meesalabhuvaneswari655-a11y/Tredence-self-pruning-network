# Self-Pruning Neural Network Report

## Objective

In this assignment, I implemented a neural network that can reduce its own unnecessary connections during training. Instead of removing weights after training, I added a learnable gate for every weight so that the network can decide which connections should remain active.

## Implementation

I created a custom `PrunableLinear` layer instead of using `torch.nn.Linear` directly. Each layer contains:

- a weight matrix,
- a bias vector,
- and a trainable `gate_scores` matrix with the same shape as the weights.

During the forward pass, I applied the sigmoid function to `gate_scores` to obtain gate values between 0 and 1. These gate values were multiplied element-wise with the weights before performing the linear transformation.

The network architecture used was:

- Input: 32 × 32 × 3 (CIFAR-10 image)
- Hidden Layer 1: 512 units
- Hidden Layer 2: 256 units
- Output Layer: 10 classes

## Loss Function

I trained the model using:

**Total Loss = CrossEntropyLoss + λ × SparsityLoss**

where `SparsityLoss` is the sum of all sigmoid gate values across the network.

## Why L1 Regularization Creates Sparsity

The sigmoid gate values are always between 0 and 1. By minimizing the sum of these gate values, the optimizer tries to reduce unnecessary gates. When a gate becomes very close to zero, the corresponding weight contributes almost nothing to the output, so that connection is effectively removed. I observed that increasing λ caused more gate values to move toward zero.
## Experimental Setup

The model was trained on the CIFAR-10 dataset for 20 epochs.

The final experiment used:

- Lambda (λ): 0.005
- Pruning threshold: 0.30
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64

## Experimental Results

| Parameter | Result |
|---|---:|
| Lambda (λ) | 0.005 |
| Training Epochs | 20 |
| Test Accuracy | 52.62% |
| Sparsity | 59.56% |
| Pruning Threshold | 0.30 |

## Observation

The model achieved 52.62% test accuracy while suppressing 59.56% of the connections according to the selected pruning threshold of 0.30.

This demonstrates that the learnable gates can suppress a substantial portion of network connections while retaining useful classification performance.

The experiment also shows the importance of selecting an appropriate sparsity regularization strength and pruning threshold. Excessive regularization can remove too many connections and negatively affect predictive performance.

## My Observation

The most interesting part of this assignment was seeing how λ controls the trade-off between accuracy and sparsity. With a very small λ, the model behaves almost like a normal neural network. With a large λ, many gates are forced close to zero, which greatly increases sparsity but also reduces classification accuracy. In my implementation, λ = 0.001 gave a reasonable balance between keeping the model accurate and removing unnecessary connections.

## Gate Distribution

I generated a histogram of the final gate values using matplotlib. Most gate values were concentrated near zero, which indicates that the model successfully learned to deactivate many connections during training.
