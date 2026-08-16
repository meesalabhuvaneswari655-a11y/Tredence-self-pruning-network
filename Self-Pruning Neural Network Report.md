# Self-Pruning Neural Network Report

## Objective

This project implements a neural network that learns to prune its own connections
*during* training, rather than as a separate post-training step. Every weight is
paired with a learnable gate; the network is trained so that unimportant gates are
driven toward zero, effectively removing the corresponding weight.

## Implementation

`PrunableLinear` (in `model.py`) replaces `torch.nn.Linear`. Each instance owns:

- a `weight` matrix (Kaiming-uniform initialized, same as `nn.Linear`'s default),
- a `bias` vector,
- a `gate_scores` matrix, same shape as `weight`, initialized to zero.

Forward pass:

```
gates          = sigmoid(gate_scores)      # in (0, 1)
pruned_weight  = weight * gates
output         = pruned_weight @ x + bias
```

Both `weight` and `gate_scores` are `nn.Parameter`s, so Adam updates both during
`backward()`. Gradients flow into `weight` through the elementwise product, and into
`gate_scores` through the sigmoid.

Network: `3072 -> 512 -> 256 -> 10` (flattened 32×32×3 CIFAR-10 image), ReLU
activations between hidden layers.

## Loss Function

```
Total Loss = CrossEntropyLoss + λ * SparsityLoss
SparsityLoss = sum of all sigmoid gate values across all PrunableLinear layers
```

## Why an L1 Penalty on the Gates Encourages Sparsity

Gate values are always non-negative (sigmoid output), so the L1 norm of the gates is
just their sum — no absolute value needed. Two properties of L1 make it specifically
good at producing *exact* zeros, as opposed to just "small" values:

1. **Constant gradient magnitude.** The derivative of `|g|` w.r.t. `g` is `±1`
   everywhere except at 0 — it doesn't shrink as `g` gets smaller. An L2 penalty
   (`g²`) has gradient `2g`, which vanishes as `g → 0`, so L2 asymptotically shrinks
   values but rarely eliminates them. L1's constant pull keeps pushing a gate down
   even once it's already small, until the classification loss's gradient (which
   *wants* to keep the gate active because that weight is useful) exceeds it.
2. **A tug-of-war per weight.** Each gate is a small optimization in itself: the
   sparsity term pushes it toward 0, the classification loss pushes it toward
   whatever value improves accuracy. Weights that barely help classification lose
   this tug-of-war quickly and their `gate_scores` get driven to large negative
   values, where sigmoid saturates near 0. Weights that matter a lot resist the pull
   and stay near 1.

Increasing λ increases the strength of the first force, so more gates lose the
tug-of-war and collapse toward zero — at the cost of also dragging down some gates
that were mildly useful, which is why accuracy drops as λ grows.

## Experimental Setup

- Optimizer: Adam, learning rate 0.001
- Batch size: 64
- Epochs: 20

## Experimental Results

The spec asks for a comparison across at least three λ values (low, medium, high) to
show the accuracy/sparsity trade-off. **So far only one run (λ=0.005) has been
completed** — the table below has placeholder rows for the low and high λ values
that still need to be run before this satisfies the assignment:

| Lambda (λ) | Test Accuracy | Sparsity @ 1e-2 | Sparsity @ 0.30 |
|------------|---------------|------------------|------------------|
| 0.0005 (Low) | 54.8%* | 0.0%* | 42.0%* |
| 0.005 (Medium) | **52.62%** | **0.0%** | **59.56%** |
| 0.02 (High) | 45.3%* | 0.0%* | 78.5%* |

\* Expected/illustrative values, not experimentally measured.

Run these with:
```
python train.py --lambdas 0.0005 0.005 0.02 --epochs 20
```

**Note on threshold:** sparsity here is reported at a pruning threshold of 0.30,
which is higher than the spec's reference example of 1e-2. A gate below 0.30 is
still fairly "closed" but not as strict a cutoff as 1e-2. Report sparsity at 1e-2
as well (run `evaluate.py` on the saved checkpoint, or extend the sweep already in
that script) so the number can be compared directly against the spec's own
definition, not just against your own choice of cutoff.

## Observation

λ controls the accuracy/sparsity trade-off: a very small λ leaves the sparsity term
negligible relative to the classification loss, so the model trains close to a
normal MLP. A large λ dominates the loss and forces most gates toward zero, which
increases sparsity but starves the network of connections it needs for accuracy.

At λ=0.005, the model reached 52.62% test accuracy with 59.56% of connections below
the 0.30 threshold. This is a real trade-off point, but it can't yet be called "the
best balance" — that claim only makes sense relative to the low- and high-λ runs
still pending. Once those are in, pick whichever λ gives the best accuracy for a
given sparsity level you're targeting (or vice versa), and say so explicitly with
numbers from the table.

## Gate Distribution

`plot_gate_distribution()` in `utils.py` histograms every gate value in the trained
network (saved to `outputs/gate_histogram_lambda_*.png`). A successful run should
show a **bimodal** distribution — a spike near 0 (pruned connections) and a separate
cluster near 1 (kept connections). If the histogram instead shows everything shifted
toward zero with no distinct cluster near 1, that's a sign λ=0.005 may already be
suppressing some gates the network would have found useful with more training time
or a smaller λ — worth checking directly in the saved plot rather than assuming.
