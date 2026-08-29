import os
import sys
import time
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

# Add src to path to allow importing core
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.linear_algebra_and_stat.core.decomposition import CustomPCA

def compare_pca_vs_svd():
    print("--- PCA vs SVD Benchmark ---")
    
    # Load dataset with high dimensionality to show PCA limitations
    print("Loading high-dimensional dataset (Olivetti Faces)...")
    faces = fetch_olivetti_faces()
    X = faces.data
    n, d = X.shape
    print(f"Dataset shape: {n} samples, {d} features")
    print(f"Note: d ({d}) >> n ({n}). Covariance matrix will be a massive {d}x{d}!\n")
    
    k = 10
    
    # 1. Custom PCA (Covariance Method)
    print("Running Custom PCA (Covariance-based)...")
    start_pca = time.perf_counter()
    pca = CustomPCA(n_components=k)
    pca.fit(X)
    Z_pca = pca.transform(X)
    X_reconstructed_pca = pca.inverse_transform(Z_pca)
    pca_time = time.perf_counter() - start_pca
    
    # 2. SVD Method Directly on Centered Data
    print("Running SVD directly...")
    start_svd = time.perf_counter()
    mu = np.mean(X, axis=0)
    X_c = X - mu
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    
    # Truncate
    Uk = U[:, :k]
    Sk = np.diag(S[:k])
    Vtk = Vt[:k, :]
    
    Z_svd = X_c @ Vtk.T
    X_reconstructed_svd = (Z_svd @ Vtk) + mu
    svd_time = time.perf_counter() - start_svd
    
    print("\n--- Results ---")
    print(f"PCA Runtime: {pca_time:.6f} seconds")
    print(f"SVD Runtime: {svd_time:.6f} seconds")
    
    # 3. Mathematical Equivalence: Eigenvalues vs Singular Values
    print("\n[Mathematical Equivalence]")
    pca_evals = pca.explained_variance_[:k]
    svd_evals_derived = (S[:k]**2) / (n - 1)
    
    print("Top 5 PCA Eigenvalues:                 ", np.round(pca_evals[:5], 2))
    print("Top 5 derived from SVD (S^2/(n-1)):    ", np.round(svd_evals_derived[:5], 2))
    print(f"Max difference in eigenvalues:         {np.max(np.abs(pca_evals - svd_evals_derived)):.2e}")
    
    # 4. Principal Directions (Eigenvectors vs Right Singular Vectors)
    print("\n[Principal Directions]")
    pca_components = pca.components_[:k]
    svd_components = Vt[:k]
    
    # Check if directions are the same (they might have opposite signs)
    # The dot product of corresponding vectors should be 1 or -1
    dot_products = [np.dot(pca_components[i], svd_components[i]) for i in range(k)]
    print(f"Dot products of corresponding components (should be ~1 or ~-1):")
    print(np.round(dot_products, 4))
    
    # 5. Reconstruction Error
    mse_pca = np.mean((X - X_reconstructed_pca)**2)
    mse_svd = np.mean((X - X_reconstructed_svd)**2)
    print("\n[Reconstruction Error (k=10)]")
    print(f"PCA MSE: {mse_pca:.4f}")
    print(f"SVD MSE: {mse_svd:.4f}")
    
    print("\nConclusion: PCA through covariance and SVD on centered data yield mathematically equivalent results.")
    print("Sign differences in principal directions are normal and expected since v and -v span the same 1D space.")

if __name__ == "__main__":
    compare_pca_vs_svd()
