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

class SVDImageCompressor:
    """
    Performs SVD compression on grayscale or color images.
    Supports channel-by-channel SVD for color images.
    """
    def __init__(self):
        self.U = None
        self.S = None
        self.Vt = None
        self.is_color = False
        self.shape = None
        
    def fit(self, image):
        """
        Decomposes the image using full-rank (thin) SVD.
        Expects a 2D array (grayscale) or 3D array (color).
        """
        image_float = np.asarray(image, dtype=np.float64)
        self.shape = image_float.shape
        self.is_color = len(image_float.shape) == 3
        
        if not self.is_color:
            self.U, self.S, self.Vt = np.linalg.svd(image_float, full_matrices=False)
        else:
            self.U, self.S, self.Vt = [], [], []
            for c in range(3):
                u, s, vt = np.linalg.svd(image_float[:, :, c], full_matrices=False)
                self.U.append(u)
                self.S.append(s)
                self.Vt.append(vt)
        return self
        
    def get_rank_1_term(self, i):
        """
        Returns the i-th rank-1 matrix: sigma_i * u_i * v_i^T
        """
        if not self.is_color:
            u_i = self.U[:, i:i+1]
            s_i = self.S[i]
            v_i_t = self.Vt[i:i+1, :]
            return s_i * (u_i @ v_i_t)
        else:
            term = np.zeros(self.shape)
            for c in range(3):
                u_i = self.U[c][:, i:i+1]
                s_i = self.S[c][i]
                v_i_t = self.Vt[c][i:i+1, :]
                term[:, :, c] = s_i * (u_i @ v_i_t)
            return term

    def compress(self, k, clip=True):
        """
        Returns the rank-k approximation of the image.
        """
        if not self.is_color:
            Uk = self.U[:, :k]
            Sk = np.diag(self.S[:k])
            Vtk = self.Vt[:k, :]
            reconstructed = Uk @ Sk @ Vtk
        else:
            reconstructed = np.zeros(self.shape)
            for c in range(3):
                Uk = self.U[c][:, :k]
                Sk = np.diag(self.S[c][:k])
                Vtk = self.Vt[c][:k, :]
                reconstructed[:, :, c] = Uk @ Sk @ Vtk
                
        if clip:
            return np.clip(reconstructed, 0, 255)
        return reconstructed
            
    def memory_stats(self, k):
        """
        Calculates storage requirements and compression ratio for rank k.
        """
        if not self.is_color:
            m, n = self.shape
            orig_size = m * n
            comp_size = k * (m + n + 1)
        else:
            m, n, _ = self.shape
            orig_size = m * n * 3
            comp_size = 3 * k * (m + n + 1)
            
        ratio = comp_size / orig_size
        return {
            "original_size": orig_size,
            "compressed_size": comp_size,
            "compression_ratio": ratio,
            "reduction_pct": (1.0 - ratio) * 100
        }
