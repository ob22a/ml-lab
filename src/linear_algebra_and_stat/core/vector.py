import math
from typing import List, Union

class Vector:
    """
    A class representing an n-dimensional vector.
    """

    def __init__(self, elements: List[Union[int, float]]):
        self.elements = list(elements)
    
    def __len__(self) -> int:
        return len(self.elements)
    
    def __getitem__(self, index: int) -> Union[int, float]:
        return self.elements[index]
    
    def __iter__(self):
        return iter(self.elements)

    def _check_dimensions(self, other: 'Vector'):
        if len(self) != len(other):
            raise ValueError(f"Vector dimensions must match. Got {len(self)} and {len(other)}.")

    def __add__(self, other: 'Vector') -> 'Vector':
        self._check_dimensions(other)
        return Vector([a + b for a, b in zip(self, other)])

    def __sub__(self, other: 'Vector') -> 'Vector':
        self._check_dimensions(other)
        return Vector([a - b for a, b in zip(self, other)])

    def __mul__(self, scalar: Union[int, float]) -> 'Vector':
        return Vector([a * scalar for a in self])

    def __rmul__(self, scalar: Union[int, float]) -> 'Vector':
        return self.__mul__(scalar)

    def __eq__(self, other: 'Vector') -> bool:
        if not isinstance(other, Vector) or len(self) != len(other):
            return False
        return all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) for a, b in zip(self, other))

    def dot(self, other: 'Vector') -> float:
        self._check_dimensions(other)
        return sum(a * b for a, b in zip(self, other))

    def norm(self, p: Union[int, float, str] = 2) -> float:
        # Either max norm, manhattan norm, euclidean norm, or p-norm
        if p == 'inf':
            return max(abs(x) for x in self)
        elif p == 1:
            return sum(abs(x) for x in self)
        elif p == 2:
            return math.sqrt(sum(x**2 for x in self))
        else:
            return sum(abs(x)**p for x in self) ** (1/p)

    def normalized(self) -> 'Vector':
        n = self.norm(2)
        if n == 0:
            raise ValueError("Cannot normalize a zero vector.")
        return self * (1.0 / n)

    def is_orthogonal(self, other: 'Vector', tol: float = 1e-9) -> bool:
        return abs(self.dot(other)) < tol

    def project_onto(self, other: 'Vector') -> 'Vector':
        """
        Projects this vector (self) onto another vector (other).
        proj_u(v) = (v·u / u·u) * u
        """
        u_dot_u = other.dot(other)
        if u_dot_u == 0:
            raise ValueError("Cannot project onto a zero vector.")
        scalar_projection = self.dot(other) / u_dot_u
        return other * scalar_projection

    def __repr__(self) -> str:
        return f"Vector {self.elements}"

if __name__ == "__main__":
    # Example usage
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    v3=Vector([1, 3, -7/3])
    
    print("v1:", v1)
    print("v2:", v2)
    print("v1 + v2:", v1 + v2)
    print("v1 - v2:", v1 - v2)
    print("Dot product:", v1.dot(v2))
    print("Norm of v1:", v1.norm())
    print("Normalized v1:", v1.normalized())
    print("Is orthogonal:", v1.is_orthogonal(v2))
    print("Is orthogonal (v1 and v3):", v1.is_orthogonal(v3))
    print("Projection of v1 onto v2:", v1.project_onto(v2))