import numpy as np
from typing import Tuple
from sklearn.decomposition import PCA

def compute_explained_variance(X: np.ndarray) -> Tuple[np.ndarray, PCA]:
    """
    Fits PCA on the dataset and returns the cumulative explained variance.
    """
    pca = PCA()
    pca.fit(X)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    return cumulative_variance, pca

def reconstruct_images(X: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
    """
    Fits PCA with k components, reconstructs the data, and computes MSE.
    """
    pca_k = PCA(n_components=k)
    Z = pca_k.fit_transform(X)
    X_reconstructed = pca_k.inverse_transform(Z)
    mse = np.mean((X - X_reconstructed) ** 2)
    return X_reconstructed, float(mse)

if __name__ == "__main__":
    from sklearn.datasets import load_digits
    
    print("Running Experiment 3: Image Dataset (Digits)")
    
    digits = load_digits()
    X = digits.data
    n, d = X.shape
    print(f"Original data shape: {n} samples (images), {d} features (pixels)")
    
    cumulative_variance, pca_full = compute_explained_variance(X)
    print(f"Total features: {d}")
    print(f"Components for 90% variance: {np.argmax(cumulative_variance >= 0.90) + 1}")
    print(f"Components for 95% variance: {np.argmax(cumulative_variance >= 0.95) + 1}")
    
    print("\n--- Reconstruction Errors ---")
    k_values = [2, 5, 10, 20, 50, 64]
    for k in k_values:
        _, mse = reconstruct_images(X, k)
        print(f"k={k:2d}: MSE={mse:.4f}")
