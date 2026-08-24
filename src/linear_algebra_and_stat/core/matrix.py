from typing import List, Union
from .vector import Vector
import math

class Matrix:
    """
    A class representing a matrix using nested Python lists.
    """

    def __init__(self, elements: List[List[Union[int, float]]]):
        if not elements or not elements[0]:
            raise ValueError("Matrix cannot be empty.")
        
        self.rows = len(elements)
        self.cols = len(elements[0])
        
        # Validate that all rows have the same number of columns
        for row in elements:
            if len(row) != self.cols:
                raise ValueError("All rows in the matrix must have the same length.")
                
        self.elements = [list(row) for row in elements]

    @property
    def shape(self) -> tuple:
        return (self.rows, self.cols)

    @property
    def is_square(self) -> bool:
        return self.rows == self.cols

    @property
    def is_symmetric(self) -> bool:
        if not self.is_square:
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                if not math.isclose(self.elements[i][j], self.elements[j][i], rel_tol=1e-9, abs_tol=1e-9):
                    return False
        return True

    def transpose(self) -> 'Matrix':
        transposed_elements = [
            [self.elements[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ]
        return Matrix(transposed_elements)

    @staticmethod
    def identity(n: int) -> 'Matrix':
        if n <= 0:
            raise ValueError("Identity matrix size must be greater than 0.")
        elements = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        return Matrix(elements)

    def __add__(self, other: 'Matrix') -> 'Matrix':
        if self.shape != other.shape:
            raise ValueError(f"Matrix dimensions must match for addition. Got {self.shape} and {other.shape}.")
        
        result_elements = [
            [self.elements[i][j] + other.elements[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(result_elements)

    def __mul__(self, other: Union['Matrix', Vector, int, float]) -> Union['Matrix', Vector]:
        if isinstance(other, (int, float)):
            # Scalar multiplication
            result_elements = [
                [self.elements[i][j] * other for j in range(self.cols)]
                for i in range(self.rows)
            ]
            return Matrix(result_elements)
            
        elif isinstance(other, Vector):
            # Matrix-Vector multiplication
            if self.cols != len(other):
                raise ValueError(f"Matrix columns ({self.cols}) must match Vector length ({len(other)}).")
            
            result_elements = [
                sum(self.elements[i][j] * other[j] for j in range(self.cols))
                for i in range(self.rows)
            ]
            return Vector(result_elements)
            
        elif isinstance(other, Matrix):
            # Matrix-Matrix multiplication
            if self.cols != other.rows:
                raise ValueError(f"Matrix 1 columns ({self.cols}) must match Matrix 2 rows ({other.rows}).")
            
            result_elements = [
                [
                    sum(self.elements[i][k] * other.elements[k][j] for k in range(self.cols))
                    for j in range(other.cols)
                ]
                for i in range(self.rows)
            ]
            return Matrix(result_elements)
        
        else:
            print(f"Multiplication with type {type(other)} is not supported.")
            return NotImplemented

    def inverse(self) -> 'Matrix':
        if not self.is_square:
            raise ValueError("Only square matrices can be inverted.")
        
        det = self.determinant()
        if math.isclose(det, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Matrix is singular and cannot be inverted.")
        
        # For 2x2 matrix, use the formula for inverse
        if self.shape == (2, 2):
            a, b = self.elements[0]
            c, d = self.elements[1]
            inv_elements = [[d/det, -b/det], [-c/det, a/det]]
            return Matrix(inv_elements)
        
        # For larger matrices, use the adjugate method
        cofactors = []
        for i in range(self.rows):
            cofactor_row = []
            for j in range(self.cols):
                minor_matrix = self.get_minor(i, j)
                cofactor_value = ((-1) ** (i + j)) * minor_matrix.determinant()
                cofactor_row.append(cofactor_value)
            cofactors.append(cofactor_row)
        
        cofactor_matrix = Matrix(cofactors)
        adjugate_matrix = cofactor_matrix.transpose()
        
        inv_elements = [
            [adjugate_matrix.elements[i][j] / det for j in range(adjugate_matrix.cols)]
            for i in range(adjugate_matrix.rows)
        ]
        
        return Matrix(inv_elements)
    
    def rank(self, tol: float = 1e-9) -> int:
        """
        Computes the rank of the matrix using Gaussian elimination to 
        convert the matrix into Row Echelon Form (REF).
        
        The rank is the number of non-zero rows in the REF.
        """
        # Create a copy of the elements as we will mutate them
        elements = [list(row) for row in self.elements]
        rows = self.rows
        cols = self.cols
        
        r = 0 # row index for pivot
        
        # Print states for debugging
        # print(f"Initial matrix elements: {elements}")
        
        for c in range(cols):
            if r >= rows:
                print("All rows have been processed. Exiting loop.")
                break
                
            # Find the row with the largest absolute value in this column (partial pivoting)
            max_val = 0
            max_row = r
            for i in range(r, rows):
                if abs(elements[i][c]) > max_val:
                    max_val = abs(elements[i][c])
                    max_row = i
                    
            # If the maximum value is essentially zero, this column has no pivot
            if max_val < tol:
                continue
                
            # Swap current row with max_row
            if max_row != r:
                elements[r], elements[max_row] = elements[max_row], elements[r]
                
            # Normalize the pivot row
            pivot = elements[r][c]
            for j in range(c, cols):
                elements[r][j] /= pivot
            
            #print(f"After processing column {c}, pivot row {r} normalized: {elements[r]}")
                
            # Eliminate the current column in the rows below
            for i in range(r + 1, rows):
                factor = elements[i][c]
                for j in range(c, cols):
                    elements[i][j] -= factor * elements[r][j]
                    
            r += 1

            # Print the each step of the elimination process for debugging
            # print(f"After processing column {c}, matrix elements: {elements}")
            
        # The rank is the number of rows with at least one non-zero element
        rank_count = 0
        for i in range(rows):
            if any(abs(val) > tol for val in elements[i]):
                rank_count += 1
                
        return rank_count
      
    def get_minor(self, row: int, col: int) -> 'Matrix':
        """
        Returns the minor matrix after removing the specified row and column.
        """
        minor_elements = [
            [self.elements[i][j] for j in range(self.cols) if j != col]
            for i in range(self.rows) if i != row
        ]
        return Matrix(minor_elements)
    
    def determinant(self) -> float:
        """
            Computes the determinant of the matrix using a recursive approach.
        """

        if not self.is_square:
            raise ValueError("Determinant can only be computed for square matrices.")
        
        # Base case for 1x1 matrix
        if self.shape == (1, 1):
            return self.elements[0][0]
        
        # Base case for 2x2 matrix
        if self.shape == (2, 2):
            a, b = self.elements[0]
            c, d = self.elements[1]
            return a * d - b * c
        
        # Recursive case for larger matrices
        det = 0
        for j in range(self.cols):
            minor_matrix = self.get_minor(0, j)
            cofactor = ((-1) ** (0 + j)) * self.elements[0][j] * minor_matrix.determinant()
            det += cofactor
            
        return det

    def eigenvalues(self) -> List[float]:
        """
        Computes the eigenvalues of the matrix.
        For 2x2 matrices, it uses the characteristic polynomial.
        For larger matrices, it raises NotImplementedError.
        """
        if self.shape != (2, 2):
            raise NotImplementedError("Eigenvalue computation is only implemented for 2x2 matrices.")
        
        a,b = self.elements[0]
        c,d = self.elements[1]

        trace = a + d
        det = a * d - b * c

        discriminant = trace**2 - 4 * det
        if discriminant < 0:
            raise ValueError("Complex eigenvalues are not supported in this basic implementation.")
        
        sqrt_discriminant = math.sqrt(discriminant)
        lambda1 = (trace + sqrt_discriminant) / 2
        lambda2 = (trace - sqrt_discriminant) / 2

        return [lambda1, lambda2]
    
    def eigenvectors(self, eigenvalues: List[float] = None, tol: float = 1e-9) -> List[Vector]:
        """
        Computes the eigenvectors corresponding to the eigenvalues of the matrix.
        For 2x2 matrices, it solves (A - λI)v = 0 for each eigenvalue λ.
        For larger matrices, it raises NotImplementedError.
        """
        if self.shape != (2, 2):
            raise NotImplementedError("Eigenvector computation is only implemented for 2x2 matrices.")
        
        if eigenvalues is None:
            eigenvalues = self.eigenvalues()
        
        a, b = self.elements[0]
        c, d = self.elements[1]
        
        eigenvectors: List[Vector] = []
        
        for lam in eigenvalues:
            # Solve (A - λI)v = 0
            a_minus_lam = a - lam
            d_minus_lam = d - lam
            
            # If b is non-zero, from (a-λ)x + by = 0 we get v = [b, -(a-λ)]
            if abs(b) > tol:
                v = Vector([b, -a_minus_lam])
            # If c is non-zero, from cx + (d-λ)y = 0 we get v = [(d-λ), -c]
            elif abs(c) > tol:
                v = Vector([d_minus_lam, -c])
            # If both b and c are zero, it's a diagonal matrix.
            else:
                if abs(a_minus_lam) < tol:
                    v = Vector([1, 0])
                else:
                    v = Vector([0, 1])
            
            eigenvectors.append(v)
        
        # Normalize eigenvectors
        try:
            return [v.normalized() for v in eigenvectors]
        except ValueError:
            print("Error: Cannot normalize a zero vector.")
            return eigenvectors
    
    def __rmul__(self, scalar: Union[int, float]) -> 'Matrix':
        return self.__mul__(scalar)

    def __eq__(self, other: 'Matrix') -> bool:
        if not isinstance(other, Matrix) or self.shape != other.shape:
            return False
        
        for i in range(self.rows):
            for j in range(self.cols):
                if not math.isclose(self.elements[i][j], other.elements[i][j], rel_tol=1e-9, abs_tol=1e-9):
                    return False
        return True

    def __repr__(self) -> str:
        rows_str = "\n".join(str(row) for row in self.elements)
        return f"Matrix {self.shape} \n{rows_str}"

if __name__ == "__main__":
    # Example usage
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    C = A + B
    D = A * B
    E = A.transpose()
    F = Matrix.identity(3)
    G = A.inverse()

    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    print("\nMatrix C (A + B):")
    print(C)
    print("\nMatrix D (A * B):")
    print(D)
    print("\nTranspose of A:")
    print(E)
    print("\nIdentity Matrix of size 3:")
    print(F)
    print("\nInverse of A:")
    print(G)
    print("\nProduct of A and its inverse (should be identity):")
    print(A * G)
    print("\nRank of A:")
    print(A.rank())