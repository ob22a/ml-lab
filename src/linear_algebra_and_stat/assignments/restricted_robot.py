from typing import Tuple, Optional
from ..core.vector import Vector

def check_robot_reachability(target_x: int, target_y: int) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    Determines if a restricted robot can reach a target integer coordinate (x, y).
    
    The robot has two allowed movement vectors:
    Move A = (2, 1)
    Move B = (1, -2)
    
    The robot can only use integer multiples of these moves.
    We need to solve for integers a and b such that:
    a(2, 1) + b(1, -2) = (target_x, target_y)
    
    Which gives the system:
    2a + b = target_x
    a - 2b = target_y
    
    By substitution or elimination, we derive:
    a = (2*target_x + target_y) / 5
    b = (target_x - 2*target_y) / 5
    
    Returns:
        (is_reachable, a, b) where a and b are the integer coefficients if reachable.
    """
    
    # Calculate analytical solutions
    a_float = (2 * target_x + target_y) / 5.0
    b_float = (target_x - 2 * target_y) / 5.0
    
    # Check if they are integers
    if a_float.is_integer() and b_float.is_integer():
        return True, int(a_float), int(b_float)
    else:
        return False, None, None

def simulate_robot_moves(target_x: int, target_y: int):
    """
    Helper function to run a simulation and print the results nicely.
    """
    reachable, a, b = check_robot_reachability(target_x, target_y)
    
    print(f"Target: ({target_x}, {target_y})")
    if reachable:
        print(f"  Status: REACHABLE")
        print(f"  Moves needed: {a} of Move A (2, 1), and {b} of Move B (1, -2)")
        
        # Verify using our Vector class
        move_a = Vector([2, 1])
        move_b = Vector([1, -2])
        result = (move_a * a) + (move_b * b)
        print(f"  Verification: {a}*{move_a} + {b}*{move_b} = {result}")
    else:
        print(f"  Status: UNREACHABLE")
        print(f"  Reason: The required coefficients are not integers.")
    print("-" * 50)

if __name__ == "__main__":
    # Test cases
    print("Running Restricted Robot Test Cases...\n")
    
    # Reachable cases
    simulate_robot_moves(5, 0)
    simulate_robot_moves(3, -1)
    
    # Origin
    simulate_robot_moves(0, 0)
    
    # Negative multiples
    simulate_robot_moves(-5, 0)
    simulate_robot_moves(-1, -3)
    
    # Unreachable coordinates
    simulate_robot_moves(1, 1)
    simulate_robot_moves(2, 2)
