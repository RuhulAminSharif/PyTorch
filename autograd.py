import math
import torch

# ==============================================================================
# 1. WHY WE NEED AUTOGRAD: The Problem with Manual Differentiation
# ==============================================================================
"""
In standard mathematics, if y = x^2, the derivative dy/dx = 2x.
Writing a function for this is trivial:
"""


def manual_dy_dx(x):
    """Calculates derivative of y = x^2 manually."""
    return 2 * x


"""
However, Deep Learning relies on heavily nested functions (Chain Rule).
Example: 
  y = x^2
  z = sin(y)
  
Chain Rule: 
  dz/dx = (dz/dy) * (dy/dx) 
        = cos(y) * 2x 
        = 2x * cos(x^2)
"""


def manual_dz_dx(x):
    """Calculates derivative of z = sin(x^2) manually."""
    return 2 * x * math.cos(x**2)


"""
As architectures grow (e.g., u = e^z, where z = sin(x^2)), manual chain rule 
calculations (du/dx = du/dz * dz/dy * dy/dx) become impossible to code and 
maintain by hand. We need an automated way to compute these gradients.
"""


# ==============================================================================
# 2. THE NEURAL NETWORK TRAINING PROCESS
# ==============================================================================
"""
A standard training loop involves:
1. Forward Pass: Compute predictions from inputs.
    z = w * x + b
    y_pred = sigmoid(z) = 1 / (1 + e^-z)

2. Calculate Loss: Measure how far predictions are from reality.
    Loss = -[y_target * ln(y_pred) + (1 - y_target) * ln(1 - y_pred)]

3. Backward Pass (Backpropagation): Compute gradients of the loss w.r.t weights.
    Trace backward: Loss -> y_pred -> z -> w, b

4. Update Weights: Adjust weights using gradients to minimize loss.
"""


# ==============================================================================
# 3. WHAT IS AUTOGRAD?
# ==============================================================================
"""
Autograd is PyTorch's automatic differentiation engine. It records all operations
performed on tensors with `requires_grad=True` into a computational graph.
When we call `.backward()` on a resulting tensor, PyTorch traverses this graph
backwards to automatically compute all gradients using the chain rule.
"""

# Example A: Simple Power Function
x = torch.tensor(3.0, requires_grad=True)
y = x**2

y.backward()  # Computes dy/dx and stores it in x.grad
print("\n--- Basic Autograd ---")
print(f"x: {x.item()}, y: {y.item()}")
print(f"dy/dx (x.grad): {x.grad.item()}")  # Should be 2 * 3.0 = 6.0


# Example B: Nested Function (Chain Rule)
x = torch.tensor(3.0, requires_grad=True)
y = x**2
z = torch.sin(y)

z.backward()  # Computes dz/dx
print(f"dz/dx: {x.grad.item()}")  # Should be 2*3 * cos(9) = -5.4668


# ==============================================================================
# 4. SINGLE PERCEPTRON: Manual vs Autograd
# ==============================================================================
def binary_cross_entropy_loss(prediction, target):
    """Calculates standard BCE loss with epsilon to prevent log(0) errors."""
    epsilon = 1e-8
    prediction = torch.clamp(prediction, epsilon, 1 - epsilon)
    return -(target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction))


# --- 4.1 Manual Calculation ---
x_val = torch.tensor(6.7)  # Input feature
y_true = torch.tensor(0.0)  # Target label (binary)
weight = torch.tensor(1.0)  # Weight
bias = torch.tensor(0.0)  # Bias

# Forward pass
z_val = weight * x_val + bias
y_pred = torch.sigmoid(z_val)
loss_val = binary_cross_entropy_loss(y_pred, y_true)

# Backward pass (Manual Chain Rule)
# 1. dLoss/dy_pred
dloss_dypred = (y_pred - y_true) / (y_pred * (1 - y_pred))
# 2. dy_pred/dz (Derivative of sigmoid)
dypred_dz = y_pred * (1 - y_pred)
# 3. dz/dw and dz/db
dz_dw = x_val
dz_db = 1.0

# Chain them together
dL_dw_manual = dloss_dypred * dypred_dz * dz_dw
dL_db_manual = dloss_dypred * dypred_dz * dz_db

print("\n--- Perceptron Gradients ---")
print(
    f"Manual Gradients  -> dL/dw: {dL_dw_manual.item():.4f}, dL/db: {dL_db_manual.item():.4f}"
)


# --- 4.2 PyTorch Autograd Calculation ---
x_tensor = torch.tensor(6.7)
y_tensor = torch.tensor(0.0)
w_tensor = torch.tensor(1.0, requires_grad=True)
b_tensor = torch.tensor(0.0, requires_grad=True)

# Forward pass (Graph is built dynamically here)
z_tensor = w_tensor * x_tensor + b_tensor
y_pred_tensor = torch.sigmoid(z_tensor)
loss_tensor = binary_cross_entropy_loss(y_pred_tensor, y_tensor)

# Backward pass (PyTorch handles the chain rule)
loss_tensor.backward()
print(
    f"Autograd Gradients-> dL/dw: {w_tensor.grad.item():.4f}, dL/db: {b_tensor.grad.item():.4f}"
)


# ==============================================================================
# 5. VECTOR GRADIENTS & REDUCTION
# ==============================================================================
"""
By default, `.backward()` expects the calling tensor to be a scalar (a single number).
If we have an array/vector of outputs, we must reduce it (e.g., sum or mean) before 
calling backward, which is exactly what loss functions do.
"""
vector_x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# y =  (x1^2+x2^2+x3^2) / 3 ==> y = (1^2 + 2^2 + 3^2) / 3
vector_y = (vector_x**2).mean()
vector_y.backward()

print("\n--- Vector Gradients ---")
print(f"Gradients of mean reduction: {vector_x.grad}")  # [2/3, 4/3, 6/3]


# ==============================================================================
# 6. CLEARING GRADIENTS (Accumulation Problem)
# ==============================================================================
"""
PyTorch ACCUMULATES gradients by default. If we run backward passes in a loop 
without clearing them, the new gradients are added to the old ones, resulting in 
incorrect values during training.
"""
print("\n--- Gradient Accumulation ---")
accum_x = torch.tensor(2.0, requires_grad=True)

for step in range(2):
    out = accum_x**2
    out.backward()
    print(f"Step {step+1} gradient: {accum_x.grad.item()} (Should be 4.0)")

    # Correct way to handle: Zero the gradients after every step!
    # Without this line, Step 2's gradient would become 8.0
    accum_x.grad.zero_()
    print(f"Gradients zeroed: {accum_x.grad.item()}")


# ==============================================================================
# 7. DISABLING GRADIENT TRACKING
# ==============================================================================
"""
Why disable it? 
Tracking history takes memory and compute. During model evaluation (inference) 
or when freezing specific layers, we don't need gradients. Disabling tracking 
saves significant resources.
"""

print("\n--- Disabling Gradient Tracking ---")
track_x = torch.tensor(2.0, requires_grad=True)

# Option 1: In-place requires_grad modification
# Useful for freezing layers during Transfer Learning
track_x.requires_grad_(False)
print(f"Option 1 (requires_grad_): {track_x.requires_grad}")

# Option 2: .detach()
# Creates a new tensor sharing the same data, but detached from the computational graph.
track_x = torch.tensor(2.0, requires_grad=True)
detached_x = track_x.detach()
print(
    f"Option 2 (detach): original requires_grad={track_x.requires_grad}, detached={detached_x.requires_grad}"
)

# Option 3: torch.no_grad() Context Manager
# The standard and most common way to wrap inference code.
track_x = torch.tensor(2.0, requires_grad=True)
with torch.no_grad():
    result = track_x**2
    # Notice result does NOT have a grad_fn attached to it
    print(f"Option 3 (no_grad): result requires_grad={result.requires_grad}")
