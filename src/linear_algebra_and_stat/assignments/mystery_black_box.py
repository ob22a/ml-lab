from ..core.matrix import Matrix
from ..core.vector import Vector

def run_mystery_box():
    """
    Simulates the Mystery Black Box assignment.
    
    Given:
    T(1, 0) = (0, 1)
    T(0, 1) = (-1, 0)
    
    The images of the basis vectors become the columns of the transformation matrix.
    Therefore, the matrix A is:
    [ 0 -1 ]
    [ 1  0 ]
    
    This is a 90-degree counterclockwise rotation.
    """
    
    A = Matrix([
        [0, -1],
        [1,  0]
    ])
    
    v = Vector([2, 1])
    print(f"Original vector v: {v}")
    
    print("\nMethod 1: Repeatedly transforming the vector")
    v1 = A * v
    print(f"T(v)   = {v1}")
    
    v2 = A * v1
    print(f"T^2(v) = {v2}")
    
    v3 = A * v2
    print(f"T^3(v) = {v3}")
    
    v4 = A * v3
    print(f"T^4(v) = {v4}")
    
    print(f"Is T^4(v) equal to v? {v4 == v}")
    
    print("\nMethod 2: Matrix composition")
    A2 = A * A
    A3 = A2 * A
    A4 = A3 * A
    
    print(f"A^4 = \n{A4}")
    
    I = Matrix.identity(2)
    print(f"Is A^4 equal to Identity matrix? {A4 == I}")

if __name__ == "__main__":
    run_mystery_box()
