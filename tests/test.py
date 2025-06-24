import numpy as np
import pytest
from src.utils import o_plus_k_operator



def test_o_plus_k_operator():
    """Test the o_plus_k_operator implementation with known cases."""
    print("Testing o_plus_k_operator...")

    # Test case 1: Simple 2x2 matrix, k=1
    A = np.array([[1, 2], [3, 4]])
    result_k1 = o_plus_k_operator(A, 1)
    I = np.eye(2)
    I_2 = np.kron(I,I)
    expected_k1 = np.kron(I,A) + np.kron(A,I)  # For k=1: A ⊗ I + I ⊗ A where I is 1x1
    print("Input Matrix:")
    print(A)
    print("Computed Result:", result_k1)
    print("Expected Result:", expected_k1)
    print(f"Test 1 (k=1): {'PASS' if np.allclose(result_k1, expected_k1) else 'FAIL'}")

    # Test case 2: k=0 should return A
    result_k0 = o_plus_k_operator(A, 0)
    print(f"Test 2 (k=0): {'PASS' if np.allclose(result_k0, A) else 'FAIL'}")

    #I \otimes A \otimes I + I\otimes I \otimes A + A \otimes I \otimes I
    # Test case 3: Verify Matrices
    result_k2 = o_plus_k_operator(A, 2)
    expected_k2 = np.kron(I,result_k1) + np.kron(A,I_2)
    print(f"Test 1 (k=1): {'PASS' if np.allclose(result_k2, expected_k2) else 'FAIL'}")

    # Test case 4: Manual verification for k=2, 1x1 matrix
    A_1x1 = np.array([[2]])
    result_1x1_k2 = o_plus_k_operator(A_1x1, 2)
    # For 1x1 matrix with k=2: A⊗I⊗I + I⊗A⊗I + I⊗I⊗A = 3*A (since I=1)
    expected_1x1_k2 = 3 * A_1x1
    print(f"Test 4 (1x1, k=2): {'PASS' if np.allclose(result_1x1_k2, expected_1x1_k2) else 'FAIL'}")
