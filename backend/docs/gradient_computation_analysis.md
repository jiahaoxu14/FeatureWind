# Detailed Report: Feature Gradient Computation in FeatureWind

## Overview

FeatureWind computes feature gradients using **PyTorch's automatic differentiation** through the dimensionality reduction process (t-SNE). The gradient computation is primarily handled by `DimReader.py:8-83` in coordination with `TangentMap.py:15-57` and a custom PyTorch-based t-SNE implementation in `tsne.py`.

## Core Architecture

### 1. **DimReader.py - ProjectionRunner Class**

The `ProjectionRunner` class (`DimReader.py:8-83`) is the central component responsible for gradient computation:

**Key Components:**
- **Input Processing**: Converts input data to PyTorch tensors with `requires_grad=True` (`DimReader.py:19`)
- **Gradient Engine**: Uses `torch.autograd.grad()` to compute partial derivatives (`DimReader.py:38-39`) 
- **Jacobian Matrix**: Stores full 2×d Jacobian matrix for each data point (`DimReader.py:33-44`)
- **Mathematical Operations**: Provides metric tensor computation and pushforward operations (`DimReader.py:59-82`)

### 2. **Two-Stage t-SNE Process**

The gradient computation uses a sophisticated two-stage approach (`DimReader.py:22-25`):

```python
# Stage 1: Base projection without gradients (computational efficiency)
with torch.no_grad():
    Y_base, params = tsne(data, 2, 999, 50, 20.0, save_params=True)

# Stage 2: Single iteration with gradients enabled
Y, params = tsne(data, no_dims=2, maxIter=1, initial_dims=50, perplexity=20.0,
                initY=params[0], initBeta=params[2], betaTries=50, initIY=params[1])
```

**Rationale**: This approach avoids expensive gradient computation during the full optimization while still capturing the final gradient information.

### 3. **Gradient Extraction Method**

For each data point, the system computes gradients using PyTorch's autograd (`DimReader.py:35-44`):

```python
for i in range(len(Y)):
    # Compute ∂Y[i,0]/∂X[i,:] and ∂Y[i,1]/∂X[i,:]
    grad_x = torch.autograd.grad(Y[i, 0], data, retain_graph=True)[0][i]
    grad_y = torch.autograd.grad(Y[i, 1], data, retain_graph=True)[0][i]
    
    # Store in Jacobian: J[2*i:2*i+2, :] = [∂x/∂features, ∂y/∂features]
    self.jacobian[2*i, :] = grad_x      # ∂x/∂(f₁, f₂, ..., fₙ)
    self.jacobian[2*i+1, :] = grad_y    # ∂y/∂(f₁, f₂, ..., fₙ)
```

**Mathematical Interpretation**: 
- `grad_x[j]` = ∂(projected_x)/∂(feature_j) - how projected x-coordinate changes w.r.t. feature j
- `grad_y[j]` = ∂(projected_y)/∂(feature_j) - how projected y-coordinate changes w.r.t. feature j

## Advanced Mathematical Operations

### 1. **Jacobian Matrix Structure** (`DimReader.py:52-57`)

The full Jacobian matrix is organized as:
```
J = [∂x₁/∂f₁  ∂x₁/∂f₂  ...  ∂x₁/∂fₘ]    # Point 1, x-coordinate gradients
    [∂y₁/∂f₁  ∂y₁/∂f₂  ...  ∂y₁/∂fₘ]    # Point 1, y-coordinate gradients  
    [∂x₂/∂f₁  ∂x₂/∂f₂  ...  ∂x₂/∂fₘ]    # Point 2, x-coordinate gradients
    [∂y₂/∂f₁  ∂y₂/∂f₂  ...  ∂y₂/∂fₘ]    # Point 2, y-coordinate gradients
    ...
```

### 2. **Pullback Metric Tensor** (`DimReader.py:59-62`)

```python
def compute_metric_tensor(self, point_idx):
    """Compute G = J^T * J - measures how distances are distorted"""
    J_i = self.get_jacobian_for_point(point_idx)  # 2×d matrix for point i
    return np.dot(J_i.T, J_i)                     # d×d metric tensor
```

**Mathematical Significance**: The metric tensor `G = J^T J` quantifies how the dimensionality reduction distorts local distances in feature space.

### 3. **Pushforward Operations** (`DimReader.py:64-82`)

```python
def pushforward_vector(self, point_idx, high_d_vector):
    """Map high-dimensional vector to 2D: v_2D = J * v_high_D"""
    J_i = self.get_jacobian_for_point(point_idx)
    return np.dot(J_i, high_d_vector)

def metric_normalized_pushforward(self, point_idx, high_d_vector):
    """Metric-aware pushforward with normalization"""
    v_2d = self.pushforward_vector(point_idx, high_d_vector)
    G = self.compute_metric_tensor(point_idx)
    metric_scale = np.sqrt(np.trace(G))
    return v_2d / metric_scale if metric_scale > 1e-10 else v_2d
```

## Integration with TangentMap.py

### Data Structure Creation (`TangentMap.py:26-31`)

```python
pt = {
    "domain": p,                           # Original high-D feature values
    "range": [0, 0],                      # 2D projection coordinates
    "tangent": np.zeros((2, m)).tolist()  # Gradient matrix: 2×m
}
```

### Gradient Population (`TangentMap.py:44-52`)

```python
for i in range(m):  # For each feature dimension
    for j in range(n):  # For each data point
        tMap[j]["range"] = base_proj[j]  # Set projection coordinates
        
        # Store gradients: outPerts[j] is 2×m matrix for point j
        tMap[j]["tangent"][0][i] = float(outPerts[j][0][i])  # ∂x/∂feature_i
        tMap[j]["tangent"][1][i] = float(outPerts[j][1][i])  # ∂y/∂feature_i
```

## PyTorch t-SNE Implementation Details

### Gradient-Aware Data Processing (`tsne.py:29`)

```python
X.requires_grad_(True)  # Enable gradient tracking through t-SNE optimization
```

### Key Algorithmic Components:

1. **Pairwise Distance Computation** (`tsne.py:31-32`): 
   - Maintains gradient flow through Euclidean distance calculations
   - Uses broadcasting for efficient computation: `D = -2XX^T + ||x||² + ||x||²^T`

2. **Probability Matrix Construction** (`tsne.py:146-150`):
   - Binary search for optimal bandwidth (β) while preserving gradients
   - Symmetrization: `P = (P + P^T) / (2n)` with early exaggeration

3. **Gradient Computation** (`tsne.py:164-166`):
   - t-SNE gradient: `∂C/∂Y = 4∑ᵢ(pᵢⱼ - qᵢⱼ) * qᵢⱼ * (yᵢ - yⱼ)`
   - PyTorch autograd tracks this through the computational graph

## Computational Efficiency Considerations

### 1. **Memory Management**
- **Jacobian Storage**: `O(2n × m)` memory for n points, m features
- **Gradient Computation**: `O(n)` per point due to `retain_graph=True`

### 2. **Computational Complexity**
- **Gradient Extraction**: `O(n × m)` autograd calls
- **Total Complexity**: `O(n² × perplexity_search + n × m × autograd_cost)`

### 3. **Optimization Strategy**
The two-stage approach reduces computational cost by ~100x:
- **Without optimization**: 999 iterations × gradient computation = expensive
- **With optimization**: 999 iterations (no grad) + 1 iteration (with grad) = efficient

## Gradient Quality and Interpretation

### 1. **Physical Meaning**
- **High Gradient Magnitude**: Feature strongly influences local embedding structure
- **Gradient Direction**: Shows how local movement in feature space maps to 2D movement
- **Gradient Coherence**: Consistent gradients across nearby points indicate stable embedding

### 2. **Mathematical Properties**
- **Continuity**: Gradients are well-defined due to t-SNE's differentiable formulation
- **Locality**: Each point's gradients reflect local neighborhood preservation
- **Scale Invariance**: Relative gradient magnitudes are more meaningful than absolute values

### 3. **Validation Metrics** (Available via DimReader methods)
- **Metric Tensor Eigenvalues**: Indicate local distortion
- **Jacobian Condition Number**: Measures numerical stability
- **Pushforward Consistency**: Tests gradient accuracy through vector mapping

## Summary

FeatureWind's gradient computation system provides a mathematically rigorous foundation for feature flow visualization through:

1. **Automatic Differentiation**: PyTorch autograd ensures accurate gradient computation
2. **Efficient Two-Stage Process**: Optimizes computational cost while preserving gradient quality
3. **Complete Jacobian Information**: Enables advanced geometric operations and analysis
4. **Robust Mathematical Framework**: Metric tensors and pushforward operations support rigorous geometric interpretation

The system successfully captures how each input feature influences the 2D embedding at each data point, enabling the rich flow field visualizations that reveal feature relationships and data structure.