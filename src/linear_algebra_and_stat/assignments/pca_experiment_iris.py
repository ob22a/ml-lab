import numpy as np
from typing import Tuple

def custom_pca_iris(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs custom PCA on Iris dataset using numpy.
    Returns:
        eigenvalues: sorted eigenvalues (1D np array)
        eigenvectors: sorted eigenvectors (2D np array)
        Z: projected data (np array)
        explained_variance_ratio: explained variance ratio (1D np array)
    """
    n, d = X.shape
    
    # Center data
    mu = np.mean(X, axis=0)
    X_c = X - mu
    
    # Covariance Matrix
    cov_matrix = (X_c.T @ X_c) / (n - 1)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Project to 2D
    k = 2
    W2 = eigenvectors[:, :k]
    Z = X_c @ W2
    
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    
    return eigenvalues, eigenvectors, Z, explained_variance_ratio

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    
    print("Running Experiment 2: Iris Dataset")
    
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    
    n, d = X.shape
    print(f"Original data shape: {n} samples, {d} features")
    
    # 2. NumPy Implementation (Custom equivalent)
    eigenvalues, eigenvectors, Z_custom, explained_variance_ratio_custom = custom_pca_iris(X)
    
    # 3. Scikit-learn Implementation
    pca = PCA(n_components=2)
    Z_sklearn = pca.fit_transform(X)
    
    # 4. Compare
    print("\n--- Custom (NumPy) PCA ---")
    print(f"Top 2 Eigenvalues: {eigenvalues[:2]}")
    print(f"Explained Variance Ratio: {explained_variance_ratio_custom[:2]}")
    print(f"Projected Sample (first 3):\n{Z_custom[:3]}")
    
    print("\n--- Scikit-Learn PCA ---")
    print(f"Explained Variance (Eigenvalues): {pca.explained_variance_}")
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Projected Sample (first 3):\n{Z_sklearn[:3]}")
