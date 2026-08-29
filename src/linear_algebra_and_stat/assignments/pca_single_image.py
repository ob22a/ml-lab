import os
import numpy as np
import urllib.request
from typing import Tuple, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DEFAULT_IMAGE_PATH = os.path.join(PROJECT_ROOT, 'data', 'test_image.jpg')

def ensure_test_image_exists(image_path: str = DEFAULT_IMAGE_PATH):
    """
    Ensures that a test image exists in the data folder.
    If it doesn't exist, downloads a standard sample B/W image.
    """
    if not os.path.exists(image_path):
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        x, y = np.meshgrid(np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))
        img = np.sin(x * 10) + np.cos(y * 10) + x**2 + y**2
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(img).save(image_path)

def load_grayscale_image(image_path: str = DEFAULT_IMAGE_PATH) -> np.ndarray:
    from PIL import Image
    ensure_test_image_exists(image_path)
    img = Image.open(image_path).convert('L') # Convert to grayscale
    return np.asarray(img, dtype=np.float64)

def perform_single_image_pca(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs PCA on a single image (low-rank approximation).
    Rows are treated as samples, columns as features.
    Returns:
        mu: Mean vector
        eigenvalues: Sorted eigenvalues of the covariance matrix
        eigenvectors: Sorted eigenvectors
        Z_full: Projected coordinates (full rank)
    """
    from src.linear_algebra_and_stat.core.decomposition import CustomPCA
    
    # Use population covariance (ddof=0) to perfectly match theoretical MSE calculation
    pca = CustomPCA(ddof=0).fit(X)
    
    mu = pca.mean_
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_.T
    Z_full = pca.transform(X)
    
    return mu, eigenvalues, eigenvectors, Z_full

def compress_and_verify(X: np.ndarray, mu: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray, Z_full: np.ndarray, k_values: List[int]):
    """
    Compresses image for different k values and returns verification metrics.
    """
    n, d = X.shape
    results = []
    
    for k in k_values:
        # Reconstruct
        W_k = eigenvectors[:, :k]
        Z_k = Z_full[:, :k]
        X_hat = (Z_k @ W_k.T) + mu
        
        # Calculate theoretical sizes
        original_size = n * d
        compressed_size = (k * d) + (n * k) + d  # eigenvectors + projections + mean
        compression_ratio = compressed_size / original_size
        size_reduction_pct = (1.0 - compression_ratio) * 100
        
        # Calculate actual MSE
        actual_mse = np.mean((X - X_hat)**2)
        
        # Calculate theoretical error (Sum of discarded eigenvalues / d)
        # Because we used division by `n` in covariance, sum of eigenvalues is total variance per row.
        # To match np.mean (which divides by n * d), we divide the eigenvalue sum by d.
        discarded_eigenvalues_sum = np.sum(eigenvalues[k:])
        theoretical_mse = discarded_eigenvalues_sum / d
        
        results.append({
            'k': k,
            'X_hat': X_hat,
            'actual_mse': actual_mse,
            'discarded_eval_sum': theoretical_mse,
            'compression_ratio': compression_ratio,
            'size_reduction_pct': size_reduction_pct,
            'compressed_size': compressed_size
        })
        
    return results

if __name__ == "__main__":
    print("Running Experiment 4: Single Image PCA (Low Rank Approximation)")
    X = load_grayscale_image()
    n, d = X.shape
    print(f"Loaded image. Original size: {n}x{d} pixels ({n*d} values).")
    
    mu, eigenvalues, eigenvectors, Z_full = perform_single_image_pca(X)
    
    k_values = [10, 50, 100]
    results = compress_and_verify(X, mu, eigenvalues, eigenvectors, Z_full, k_values)
    
    for res in results:
        print(f"\n--- k = {res['k']} ---")
        print(f"Compressed size: {res['compressed_size']} values")
        print(f"Size Reduction: {res['size_reduction_pct']:.2f}%")
        print(f"Actual MSE: {res['actual_mse']:.6f}")
        print(f"Sum of discarded eigenvalues: {res['discarded_eval_sum']:.6f}")
        print(f"Difference: {abs(res['actual_mse'] - res['discarded_eval_sum']):.6e}")
