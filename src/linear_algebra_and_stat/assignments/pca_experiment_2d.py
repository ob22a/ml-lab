import numpy as np
from typing import Tuple

from ..core.matrix import Matrix

def generate_synthetic_2d_data(n_samples: int = 200) -> np.ndarray:
    """Generates an elongated, rotated 2D dataset with correlated features."""
    np.random.seed(42)
    # Generate points from a normal distribution
    # Variance in x is much larger than variance in y to make it elongated
    X = np.random.randn(n_samples, 2)
    X[:, 0] *= 3.0  # Stretch along x-axis
    X[:, 1] *= 1.0  # Keep y-axis variance small
    
    # Rotate by 45 degrees to create correlation
    theta = np.pi / 4
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    X_rotated = X @ rotation_matrix.T
    
    # Translate to move away from origin
    X_translated = X_rotated + np.array([5.0, 3.0])
    
    return X_translated

def custom_pca_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs PCA manually using the custom Matrix class for 2D data.
    Returns:
        mu: mean vector (1D np array)
        eigenvalues: sorted eigenvalues (1D np array)
        eigenvectors: sorted eigenvectors (2D np array)
        Z: projected data (np array)
        X_hat: reconstructed data (np array)
    """
    n, d = X.shape
    if d != 2:
        raise ValueError("custom_pca_2d only works for 2D data due to Matrix class limits.")
        
    # 1. Compute Mean
    mu = np.mean(X, axis=0)
    
    # 2. Center Data
    X_c = X - mu
    
    # 3. Covariance Matrix
    cov_numpy = (X_c.T @ X_c) / (n - 1)
    cov_matrix = Matrix(cov_numpy.tolist())
    
    # 4. Eigendecomposition using custom Matrix class
    eigenvalues = cov_matrix.eigenvalues()
    eigenvectors = cov_matrix.eigenvectors(eigenvalues)
    
    # Convert back to numpy for sorting and remaining steps
    evals = np.array(eigenvalues)
    evecs = np.array([v.elements for v in eigenvectors]).T
    
    # 5. Sort from largest to smallest
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    # 6. Projection
    Z = X_c @ evecs
    
    # 7. Reconstruction (using only top 1 component for demonstration)
    W1 = evecs[:, :1]
    Z1 = X_c @ W1
    X_hat = (Z1 @ W1.T) + mu
    
    return mu, evals, evecs, Z, X_hat

if __name__ == "__main__":
    print("Running Experiment 1: Synthetic 2D Data")
    X = generate_synthetic_2d_data(200)
    mu, evals, evecs, Z, X_hat = custom_pca_2d(X)
    
    print("\n--- Detailed Results ---")
    print(f"Data Shape: {X.shape}")
    print(f"Mean vector:\n{mu}")
    print(f"Eigenvalues:\n{evals}")
    print(f"Eigenvectors:\n{evecs}")
    
    print("\nOriginal Data Sample (first 3):")
    print(X[:3])
    
    print("\nProjected Data Z (first 3):")
    print(Z[:3])
    
    print("\nReconstructed Data (first 3, k=1):")
    print(X_hat[:3])
    
    print("\nReconstruction MSE (k=1):")
    print(np.mean((X - X_hat)**2))
