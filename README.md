# Deep Learning From Scratch — Day 1

## Goal

Understand the forward pass of a single artificial neuron by implementing it from scratch using NumPy.

## What I implemented

The forward pass follows this sequence:

```text
Input Data
    ↓
Weighted Sum
    ↓
z = XW + b
    ↓
Sigmoid Activation
    ↓
ŷ = sigmoid(z)
    ↓
Binary Cross-Entropy Loss
    ↓
Loss
```

## Concepts covered

* Input features and binary labels
* Weights and bias
* Linear transformation
* Sigmoid activation function
* Binary Cross-Entropy (BCE) loss
* NumPy matrix operations

## Implementation

The complete implementation is in:

```text
forward_pass.py
```

## Next Step

Implement the gradients and understand how backpropagation calculates how each parameter contributed to the loss.

