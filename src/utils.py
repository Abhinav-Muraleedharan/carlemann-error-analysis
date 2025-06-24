import numpy as np

def kronecker_power(A: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the k-th Kronecker power of matrix A: A^⊗k = A ⊗ A ⊗ ... ⊗ A (k times)

    Args:
        A: Input matrix
        k: Power (number of Kronecker products)

    Returns:
        A^⊗k
    """
    if k == 0:
        return np.array([[1.0]])  # Scalar 1
    elif k == 1:
        return A
    else:
        result = A
        for _ in range(k - 1):
            result = np.kron(result, A)
        return result


def o_plus_k_operator(A: np.ndarray, k: int) -> np.ndarray:
    """
    The o_plus_k_operator is defined as follows:

    o_plus_k(A) = A ⊗ I^⊗k + Σᵢ₌₁ᵏ⁻¹ (I^⊗i ⊗ A ⊗ I^⊗(k-i)) + I^⊗k ⊗ A

    This operator is fundamental in Carleman linearization theory for computing
    the linearized dynamics of polynomial systems.

    Args:
        A: Input matrix (n × n)
        k: Order parameter

    Returns:
        The o_plus_k operator applied to A

    Mathematical interpretation:
    - First term: A ⊗ I^⊗k acts on the first component
    - Middle terms: A acts on the (i+1)-th component for i = 1, ..., k-1
    - Last term: I^⊗k ⊗ A acts on the last component
    """
    n = A.shape[0]
    # if A.shape[1] != n:
    #     raise ValueError("Matrix A must be square")

    if k == 0:
        return A

    # Identity matrix of same size as A
    I = np.eye(n)

    # Compute I^⊗k (k-th Kronecker power of identity)
    I_k = kronecker_power(I, k)

    # Initialize result with first term: A ⊗ I^⊗k
    result = np.kron(A, I_k)

    # Add middle terms: Σᵢ₌₁ᵏ⁻¹ (I^⊗i ⊗ A ⊗ I^⊗(k-i))
    for i in range(1, k):
        I_i = kronecker_power(I, i)        # I^⊗i
        I_k_minus_i = kronecker_power(I, k - i)  # I^⊗(k-i)

        # Compute I^⊗i ⊗ A ⊗ I^⊗(k-i)
        term = np.kron(np.kron(I_i, A), I_k_minus_i)
        result += term

    # Add last term: I^⊗k ⊗ A
    result += np.kron(I_k, A)

    return result