import numpy as np

class CustomPCA:
    """
    Custom PCA implementation utilizing covariance matrix eigendecomposition.
    Built to handle N-dimensional data directly using NumPy.
    """
    def __init__(self, n_components=None, ddof=1):
        self.n_components = n_components
        self.ddof = ddof
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        # 1. Compute mean and center data
        self.mean_ = np.mean(X, axis=0)
        X_c = X - self.mean_
        n, d = X.shape
        
        # 2. Covariance matrix
        # Uses n-ddof (sample vs population covariance)
        cov_matrix = (X_c.T @ X_c) / (n - self.ddof)
        
        # 3. Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 4. Sort from largest to smallest eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.explained_variance_ = eigenvalues
        self.explained_variance_ratio_ = eigenvalues / np.sum(eigenvalues)
        self.components_ = eigenvectors.T # rows are principal components (right singular vectors)
        
        # Keep only n_components if specified
        if self.n_components is not None:
            self.explained_variance_ = self.explained_variance_[:self.n_components]
            self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]
            self.components_ = self.components_[:self.n_components]
            
        return self

    def transform(self, X):
        X_c = X - self.mean_
        return X_c @ self.components_.T

    def inverse_transform(self, Z):
        return (Z @ self.components_) + self.mean_

