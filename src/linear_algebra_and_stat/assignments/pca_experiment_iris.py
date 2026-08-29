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
    from src.linear_algebra_and_stat.core.decomposition import CustomPCA
    
    # Fit full PCA to get all eigenvalues and eigenvectors
    pca_full = CustomPCA(ddof=1).fit(X)
    
    eigenvalues = pca_full.explained_variance_
    eigenvectors = pca_full.components_.T
    Z = pca_full.transform(X)[:, :2]
    explained_variance_ratio = pca_full.explained_variance_ratio_
    
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
