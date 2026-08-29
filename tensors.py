import numpy as np
import torch
import time

# ==========================================
# 0. Environment Setup & Device Checking
# ==========================================
print(f"PyTorch Version: {torch.__version__}")

# Check for hardware accelerator (GPU) availability
if torch.cuda.is_available():
    print("GPU is available")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("GPU not available. Using CPU")
    device = torch.device("cpu")


# ==========================================
# 1. Basic Tensor Creation Methods
# ==========================================

# torch.empty: Allocates memory for a tensor without initializing values (contains garbage/memory values)
uninitialized_tensor = torch.empty(2, 3)
print("\nUninitialized Tensor:\n", uninitialized_tensor)

# torch.zeros / torch.ones: Creates tensors filled with 0s or 1s
zeros_tensor = torch.zeros(2, 3)
ones_tensor = torch.ones(2, 3)

# torch.rand: Creates a tensor with random values from a uniform distribution [0, 1)
uniform_random_tensor = torch.rand(2, 3)

# torch.manual_seed: Ensures reproducibility for random tensor generation
torch.manual_seed(100)
seeded_random_tensor = torch.rand(2, 3)

# torch.tensor: Creates a tensor explicitly from a Python list or nested array
custom_data_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# torch.arange: Generates a 1D sequence of numbers (start, end-exclusive, step)
sequential_tensor = torch.arange(0, 10)  # default step: 1
sequential_tensor = torch.arange(0, 10, 2)

# torch.linspace: Generates a 1D tensor with linearly spaced points between a start and end value
linear_spaced_tensor = torch.linspace(0, 10, 5)  # 5 points between 0 and 10

# torch.eye: Generates a 2D identity matrix (1s on the diagonal, 0s elsewhere)
identity_matrix_tensor = torch.eye(3)

# torch.full: Creates a tensor of a specific shape filled entirely with a given scalar
filled_tensor = torch.full((3, 3), 5)


# ==========================================
# 2. Tensor Shapes and '_like' Operations
# ==========================================
sample_matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Accessing dimensions
print("\nSample Matrix Shape:", sample_matrix.shape)
print("Rows (dim 0):", sample_matrix.shape[0])
print("Cols (dim 1):", sample_matrix.shape[1])

# *_like operations: Create new tensors that inherit the shape and device of a reference tensor
empty_like_tensor = torch.empty_like(sample_matrix)
zeros_like_tensor = torch.zeros_like(sample_matrix)
ones_like_tensor = torch.ones_like(sample_matrix)
rand_like_tensor = torch.rand_like(sample_matrix, dtype=torch.float32)


# ==========================================
# 3. Tensor Data Types and Transformations
# ==========================================
print("\nOriginal Data Type:", sample_matrix.dtype)

# Explicitly defining types during creation to optimize memory
int8_tensor = torch.tensor([[1.0, 2.5], [4.1, 5.9]], dtype=torch.int8)
float16_tensor = torch.tensor([[1, 2], [4, 5]], dtype=torch.float16)

# .to() method: Safely casts a tensor to a new data type
casted_matrix = sample_matrix.to(torch.float16)


# ==========================================
# 4. Mathematical & Reduction Operations
# ==========================================

# --- 4.1 Scalar Operations ---
scalar_matrix = torch.rand(2, 2, dtype=torch.float32)
print("\nOriginal Scalar Test Matrix:\n", scalar_matrix)

scalar_matrix += 2  # Addition
scalar_matrix -= 2  # Subtraction
scalar_matrix *= 3  # Multiplication
scalar_matrix /= 3  # Division
scalar_matrix = (
    scalar_matrix * 100
) // 3  # Integer Division (floor division after scaling)
scalar_matrix **= 2  # Power
scalar_matrix %= 2  # Modulo

# --- 4.2 Element-wise Operations ---
tensor_a = torch.rand(2, 3)
tensor_b = torch.rand(2, 3)

# Arithmetic applies element-by-element
elementwise_sum = tensor_a + tensor_b  # addition
elementwise_sub = tensor_a - tensor_b  # substractions
elementwise_mul = tensor_a * tensor_b  # multiplications
elementwise_div = tensor_a / tensor_b  # division
elementwise_pow = tensor_a**2  # power
elementwise_mod = tensor_a % tensor_b  # modulor

# Special mathematical functions (Absolute, Negative)
signed_tensor = torch.tensor([1, -2, 3, -4])
print("\nAbsolute Values:", torch.abs(signed_tensor))
print("Negated Values:", torch.neg(signed_tensor))

# Rounding & clamping constraints
decimal_tensor = torch.tensor([1.9, 2.3, 3.7, 4.4])
rounded_tensor = torch.round(decimal_tensor)
ceiled_tensor = torch.ceil(decimal_tensor)
floored_tensor = torch.floor(decimal_tensor)
clamped_tensor = torch.clamp(
    decimal_tensor, min=2, max=4
)  # Forces all values into the [2, 4] range

# --- 4.2 Reduction Operations ---
reduction_matrix = torch.randint(size=(2, 3), low=0, high=10, dtype=torch.float32)
print("\nReduction Matrix:\n", reduction_matrix)

# Sum operations
print("Sum (total):", torch.sum(reduction_matrix))
print("Sum (dim=0):", torch.sum(reduction_matrix, dim=0))
print("Sum (dim=1):", torch.sum(reduction_matrix, dim=1))

# Mean operations
print("Mean (total):", torch.mean(reduction_matrix))
print("Mean (dim=0):", torch.mean(reduction_matrix, dim=0))
print("Mean (dim=1):", torch.mean(reduction_matrix, dim=1))

# Max operations
print("Max (total):", torch.max(reduction_matrix))
print("Max (dim=0):", torch.max(reduction_matrix, dim=0))
print("Max (dim=1):", torch.max(reduction_matrix, dim=1))

# Min operations
print("Min (total):", torch.min(reduction_matrix))
print("Min (dim=0):", torch.min(reduction_matrix, dim=0))
print("Min (dim=1):", torch.min(reduction_matrix, dim=1))

# Product operations
print("Product (total):", torch.prod(reduction_matrix))
print("Product (dim=0):", torch.prod(reduction_matrix, dim=0))
print("Product (dim=1):", torch.prod(reduction_matrix, dim=1))

# Statistical metrics
print("Standard Deviation:", torch.std(reduction_matrix))
print("Variance:", torch.var(reduction_matrix))

# Argmax/Argmin return the *index* of the max/min value, not the value itself
max_index_flattened = torch.argmax(reduction_matrix)


# ==========================================
# 5. Matrix Operations (Linear Algebra)
# ==========================================
matrix_x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
matrix_y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# Matrix Multiplication (Dot Product)
dot_product_result = torch.matmul(matrix_x, matrix_y)
# dot_product_result = matrix_x @ matrix_y # alternate syntax
print("\nMatrix Multiplication:\n", dot_product_result)

# Transposition (Swapping rows and columns)
transposed_x = matrix_x.T

# Determinant (Requires a square matrix of floats)
det_x = torch.det(matrix_x)

# Matrix Inverse (Requires a square matrix with a non-zero determinant)
inverse_matrix = torch.inverse(matrix_x)


# ==========================================
# 6. Shape & Dimensionality Manipulation
# ==========================================
sequence_tensor = torch.arange(12)

# Reshape: Reorganizes data into a new dimensional structure (must maintain total element count)
reshaped_matrix = sequence_tensor.reshape(3, 4)

# Flatten: Collapses any N-dimensional tensor into a 1D sequence
flattened_matrix = reshaped_matrix.flatten()

# Permute: Reorders specific dimensions (common in image processing e.g., CHW to HWC)
img_tensor = torch.rand(3, 256, 256)  # (Channels, Height, Width)
permuted_img = img_tensor.permute(1, 2, 0)  # Now (Height, Width, Channels)

# Unsqueeze: Adds a dimension of size 1 at the specified index (useful for adding batch dimensions)
single_item = torch.tensor([1, 2, 3])  # Shape: (3,)
batched_item = single_item.unsqueeze(0)  # Shape: (1, 3)

# Squeeze: Removes all dimensions of size 1
squeezed_item = batched_item.squeeze()  # Shape: (3,) back to original


# ==========================================
# 7. Comparison Operations
# ==========================================
tensor_i = torch.randint(size=(2, 3), low=0, high=10)
tensor_j = torch.randint(size=(2, 3), low=0, high=10)

# Returns boolean tensors of the same shape indicating True/False per element
print("\nTensor i:\n", tensor_i)
print("Tensor j:\n", tensor_j)
print("i > j:\n", tensor_i > tensor_j)
print("i < j:\n", tensor_i < tensor_j)
print("i == j:\n", tensor_i == tensor_j)
print("i != j:\n", tensor_i != tensor_j)
print("i >= j:\n", tensor_i >= tensor_j)
print("i <= j:\n", tensor_i <= tensor_j)


# ==========================================
# 8. Special Math & Activation Functions
# ==========================================
features = torch.randn(2, 3)  # Random normal distribution (includes negatives)

log_vals = torch.log(torch.abs(features))  # Natural log (requires positive numbers)
exp_vals = torch.exp(features)  # e^x

# Deep Learning Activations
sigmoid_vals = torch.sigmoid(features)  # Squashes all values between 0 and 1
relu_vals = torch.relu(features)  # max(0, x) - replaces negatives with 0
softmax_vals = torch.softmax(
    features, dim=1
)  # Converts raw scores to probabilities that sum to 1

tensor_k = torch.randint(size=(2, 3), low=1, high=10, dtype=torch.float32)

print("\nLogarithm:\n", torch.log(tensor_k))
print("Exponential:\n", torch.exp(tensor_k))
print("Square Root:\n", torch.sqrt(tensor_k))
print("Sigmoid:\n", torch.sigmoid(tensor_k))
print("Softmax (dim=0):\n", torch.softmax(tensor_k, dim=0))
print("ReLU:\n", torch.relu(tensor_k))

# ==========================================
# 9. In-place Operations
# ==========================================
# Operations ending with an underscore (_) modify the tensor directly in memory, saving RAM
base_tensor = torch.rand(2, 3)
add_tensor = torch.ones(2, 3)

base_tensor.add_(
    add_tensor
)  # base_tensor is now updated. Same as base_tensor = base_tensor + add_tensor
base_tensor.relu_()  # Applies ReLU directly to base_tensor


# ==========================================
# 10. Memory Copying vs Cloning
# ==========================================
original_tensor = torch.rand(2, 3)

# Shallow Copy (Reference): Both variables point to the exact same memory address (use id to verify)
reference_tensor = original_tensor
original_tensor[0][0] = 99.9
# 'reference_tensor[0][0]' is also now 99.9
print(id(original_tensor))
print(id(reference_tensor))

# Deep Copy (Clone): Creates a completely independent physical copy in memory
cloned_tensor = original_tensor.clone()
original_tensor[0][0] = 42.0
# 'cloned_tensor[0][0]' remains 99.9
print(id(original_tensor))
print(id(cloned_tensor))

# ==========================================
# 11. Interoperability with NumPy
# ==========================================
# Converting PyTorch Tensor -> NumPy Array (shares underlying memory if on CPU)
torch_cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
numpy_array_from_torch = torch_cpu_tensor.numpy()

# Converting NumPy Array -> PyTorch Tensor
standard_numpy_array = np.array([4.0, 5.0, 6.0])
torch_tensor_from_numpy = torch.from_numpy(standard_numpy_array)


# ==========================================
# 12. GPU Operations & Performance Benchmarking
# ==========================================
if torch.cuda.is_available():
    benchmark_size = 10000

    matrix_cpu1 = torch.randn(benchmark_size, benchmark_size)
    matrix_cpu2 = torch.randn(benchmark_size, benchmark_size)

    # Measure CPU Time
    start_time = time.time()
    res_cpu = torch.matmul(matrix_cpu1, matrix_cpu2)
    cpu_time = time.time() - start_time
    print(f"\nTime on CPU: {cpu_time:.4f} seconds")

    # Measure GPU Time
    matrix_gpu1 = matrix_cpu1.to(device)
    matrix_gpu2 = matrix_cpu2.to(device)

    start_time = time.time()
    res_gpu = torch.matmul(matrix_gpu1, matrix_gpu2)
    gpu_time = time.time() - start_time
    print(f"Time on GPU: {gpu_time:.4f} seconds")

    print(f"Speedup multiplier: {cpu_time/gpu_time:.2f}x faster on GPU")
