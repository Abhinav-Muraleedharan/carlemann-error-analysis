import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from itertools import combinations_with_replacement, product
from src.utils import o_plus_k_operator
import time

class CarlemanLinearization:
    def __init__(self, A1, A2, n_states, truncation_order):
        """
        Initialize Carleman linearization for system: dx/dt = A1*x + A2*(x⊗x)

        Parameters:
        A1: Linear coefficient matrix (n×n)
        A2: Quadratic coefficient matrix (n×n²)
        n_states: Number of original states
        truncation_order: Maximum order for Kronecker products
        """
        self.A1 = np.array(A1)
        self.A2 = np.array(A2)
        self.n = n_states
        self.k = truncation_order

        # Validate input dimensions
        assert self.A1.shape == (self.n, self.n), f"A1 must be {self.n}×{self.n}"
        assert self.A2.shape == (self.n, self.n**2), f"A2 must be {self.n}×{self.n**2}"

        # Build the Carleman matrix Q
        self.Q = self._build_carleman_matrix_bidiagonal()

    def _kronecker_indices(self, order):
        """Generate indices for Kronecker products of given order"""
        # For Kronecker products, we need all combinations, not just combinations with replacement
        # x⊗x for 2D system should be [x1⊗x1, x1⊗x2, x2⊗x1, x2⊗x2] = 4 elements
        from itertools import product
        return list(product(range(self.n), repeat=order))

    def _kronecker_product_recursive(self, x, order):
        """Compute x^⊗order efficiently"""
        if order == 1:
            return x
        elif order == 2:
            return np.kron(x, x)
        else:
            return np.kron(x, self._kronecker_product_recursive(x, order-1))

    def _build_carleman_matrix_bidiagonal(self) -> np.ndarray:
        """
        Build the Carleman matrix with the correct bidiagonal block structure.

        The matrix has the form:
        C = [[A₁,           A₂,           0,            0,     ...]
            [0,            A₁⊕₁A₁,      A₂⊕₁A₂,      0,     ...]
            [0,            0,            A₁⊕₂A₁,      A₂⊕₂A₂, ...]
            [0,            0,            0,            A₁⊕₃A₁, ...]
            [...]]

        Args:
            A_1: First matrix operator
            A_2: Second matrix operator
            truncation_order: Order at which to truncate the Carleman matrix

        Returns:
            Carleman matrix of size (carleman_dim × carleman_dim)
        """
        # Get dimension n from A_1 (assuming A_1 is n×n)
        n = self.A1.shape[0]
        truncation_order = self.k

        # Compute Carleman dimension: n + n² + ... + n^k
        carleman_dim = sum(n**i for i in range(1, truncation_order + 1))

        # Compute order start indices and dimensions
        order_dimensions = [n**i for i in range(1, truncation_order + 1)]
        order_start_indices = [0]
        for i in range(1, truncation_order):
            order_start_indices.append(order_start_indices[-1] + order_dimensions[i-1])

        C = np.zeros((carleman_dim, carleman_dim))

        # Fill the bidiagonal blocks
        for k in range(truncation_order):
            row_start = order_start_indices[k]
            row_end = row_start + order_dimensions[k]

            # Diagonal block: A₁⊕ₖA₁ (for k=0, this is just A₁)
            col_start = row_start
            col_end = row_end

            if k == 0:
                # First diagonal block is just A₁
                C[row_start:row_end, col_start:col_end] = self.A1
            else:
                # Higher order diagonal blocks: A₁⊕ₖA₁
                A1_oplus_k = o_plus_k_operator(self.A1, k)
                C[row_start:row_end, col_start:col_end] = A1_oplus_k

            # Super-diagonal block: A₂⊕ₖA₂ (if not the last block)
            if k < truncation_order - 1:
                col_start = order_start_indices[k + 1]
                col_end = col_start + order_dimensions[k + 1]

                if k == 0:
                    # First super-diagonal block is just A₂
                    C[row_start:row_end, col_start:col_end] = self.A2
                else:
                    # Higher order super-diagonal blocks: A₂⊕ₖA₂
                    A2_oplus_k = o_plus_k_operator(self.A2, k)
                    C[row_start:row_end, col_start:col_end] = A2_oplus_k

        return C

    def _construct_carleman_matrix(self):
        """Construct the Carleman linearization matrix Q"""
        # Calculate dimensions for each order
        dims = []
        total_dim = 0
        for i in range(1, self.k + 1):
            dim_i = self.n ** i  # Correct dimension for Kronecker products
            dims.append(dim_i)
            total_dim += dim_i

        print(f"Carleman matrix dimensions: {total_dim}×{total_dim}")
        print(f"Dimension breakdown: {dims}")
        Q = np.zeros((total_dim, total_dim))

        # The Carleman matrix has a specific block structure:
        # Q = [Q11  Q12  Q13  ...]
        #     [Q21  Q22  Q23  ...]
        #     [Q31  Q32  Q33  ...]
        #     [...  ...  ...  ...]

        # Fill Q block by block
        row_start = 0
        for i in range(1, self.k + 1):
            row_end = row_start + dims[i-1]
            col_start = 0

            for j in range(1, self.k + 1):
                col_end = col_start + dims[j-1]

                # Determine what goes in block Q[i,j]
                if i == 1 and j == 1:
                    # Q_{1,1} = A1: how x evolves from linear terms
                    Q[row_start:row_end, col_start:col_end] = self.A1

                elif i == 1 and j == 2:
                    # Q_{1,2} = A2: how x evolves from quadratic terms x⊗x
                    Q[row_start:row_end, col_start:col_end] = self.A2

                elif i == 2 and j == 1:
                    # Q_{2,1}: how x⊗x evolves from linear x terms
                    # d/dt(x⊗x) = dx/dt ⊗ x + x ⊗ dx/dt = (A1⊗I + I⊗A1)(x⊗x) when dx/dt = A1*x
                    # But we want the coupling from x to x⊗x, which is more complex
                    I = np.eye(self.n)
                    Q[row_start:row_end, col_start:col_end] = self._compute_x_to_xx_coupling()

                elif i == 2 and j == 2:
                    # Q_{2,2}: how x⊗x evolves from x⊗x terms
                    I = np.eye(self.n)
                    Q[row_start:row_end, col_start:col_end] = np.kron(self.A1, I) + np.kron(I, self.A1)

                # For higher order blocks, use simplified structure
                elif i > 2 and j == 1:
                    # Higher order coupling from x
                    Q[row_start:row_end, col_start:col_end] = self._compute_higher_order_coupling(i, 1)

                elif i > 2 and j == 2:
                    # Higher order coupling from x⊗x
                    Q[row_start:row_end, col_start:col_end] = self._compute_higher_order_coupling(i, 2)

                elif i > 2 and j > 2 and i == j:
                    # Diagonal blocks for higher orders (simplified)
                    Q[row_start:row_end, col_start:col_end] = self._compute_diagonal_block(i)

                col_start = col_end
            row_start = row_end

        return Q



    def construct_augmented_state(self, x):
        """Construct augmented state vector y = [x, x⊗x, x⊗x⊗x, ...]"""
        y = x.copy()
        for order in range(2, self.k + 1):
            x_kron = self._kronecker_product_recursive(x, order)
            y = np.concatenate([y, x_kron])
        return y

    def extract_original_state(self, y):
        """Extract original state x from augmented state y"""
        return y[:self.n]

    def carleman_dynamics(self, t, y):
        """Carleman linearized dynamics: dy/dt = Q*y"""
        return self.Q @ y

    def original_dynamics(self, t, x):
        """Original nonlinear dynamics: dx/dt = A1*x + A2*(x⊗x)"""
        x_kron_2 = np.kron(x, x)
        return self.A1 @ x + self.A2 @ x_kron_2

def simulate_and_compare(A1, A2, x0, t_span, t_eval, truncation_order=3):
    """
    Simulate both original and Carleman linearized systems and compare results
    """
    n_states = len(x0)

    # Create Carleman linearization
    carleman = CarlemanLinearization(A1, A2, n_states, truncation_order)

    # Initial conditions
    y0 = carleman.construct_augmented_state(x0)

    print(f"Original system dimension: {n_states}")
    print(f"Carleman system dimension: {len(y0)}")

    # Solve original system
    print("Solving original nonlinear system...")
    sol_original = solve_ivp(carleman.original_dynamics, t_span, x0,
                           t_eval=t_eval, method='Radau', rtol=1e-10, atol=1e-12)
    #sol_original = solve_ivp(carleman.original_dynamics, t_span, x0, method='DOP853', rtol=1e-10, atol=1e-12)

    # Solve Carleman linearized system
    print("Solving Carleman linearized system...")
    sol_carleman = solve_ivp(carleman.carleman_dynamics, t_span, y0,
                           t_eval=t_eval, method='Radau', rtol=1e-10, atol=1e-12)

    # Extract original states from Carleman solution
    x_carleman = np.array([carleman.extract_original_state(y)
                          for y in sol_carleman.y.T]).T

    print("Comparison between original and Carleman:")
    print("Original Solution:")
    print(sol_original)
    print("Carleman Solution:")
    print(x_carleman)

    return sol_original, x_carleman, carleman

def plot_results(t_eval, x_original, x_carleman, truncation_order):
    """Plot comparison between original and Carleman approximation"""
    n_states = x_original.shape[0]

    fig, axes = plt.subplots(2, n_states, figsize=(4*n_states, 8))
    if n_states == 1:
        axes = axes.reshape(2, 1)

    # Plot state trajectories
    for i in range(n_states):
        axes[0, i].plot(t_eval, x_original[i, :], 'b-', label='Original', linewidth=2)
        axes[0, i].plot(t_eval, x_carleman[i, :], 'r--', label=f'Carleman (k={truncation_order})', linewidth=2)
        axes[0, i].set_xlabel('Time')
        axes[0, i].set_ylabel(f'x_{i+1}(t)')
        axes[0, i].set_title(f'State {i+1} Comparison')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)

    # Plot errors
    errors = np.abs(x_original - x_carleman)
    for i in range(n_states):
        axes[1, i].semilogy(t_eval, errors[i, :], 'g-', linewidth=2)
        axes[1, i].set_xlabel('Time')
        axes[1, i].set_ylabel(f'|Error| in x_{i+1}')
        axes[1, i].set_title(f'Absolute Error for State {i+1}')
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print error statistics
    print(f"\nError Statistics (Truncation order k={truncation_order}):")
    print("-" * 50)
    for i in range(n_states):
        max_error = np.max(errors[i, :])
        mean_error = np.mean(errors[i, :])
        final_error = errors[i, -1]
        print(f"State {i+1}: Max error = {max_error:.2e}, Mean error = {mean_error:.2e}, Final error = {final_error:.2e}")

# Example usage
if __name__ == "__main__":
    # Example 1: 2D system with Van der Pol-like dynamics
    print("Example 1: 2D Nonlinear System")
    print("=" * 40)

    # System: dx1/dt = x2 + 0.1*x1*x2
    #         dx2/dt = -x1 + 0.05*(x1^2 - x2^2)
    A1 = np.array([[-0.000, 8],
                   [-8, -0.000]])

    # A2 matrix for quadratic terms [x1*x1, x1*x2, x2*x1, x2*x2]
    A2 = np.array([[0.1, 0.0, 0, 0],
                   [0.00, 0.1, 0, 0.00]])

    x0 = np.array([0.333, 0.333])
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 200)

    sol_orig, x_carl, carleman_sys = simulate_and_compare(
        A1, A2, x0, t_span, t_eval, truncation_order=4)

    plot_results(t_eval, sol_orig.y, x_carl, 12)

    # Example 2: Test different truncation orders
    print("\nExample 2: Truncation Order Comparison")
    print("=" * 40)

    truncation_orders = [2, 3, 4,5,6,7,8,9,10]
    final_errors = []

    for k in truncation_orders:
        print(f"\nTesting truncation order k={k}")
        sol_orig, x_carl, _ = simulate_and_compare(
            A1, A2, x0, t_span, t_eval, truncation_order=k)
        print("Original Solution:")
        print(sol_orig.y[:, -1])
        print("Carleman Solution:")
        print(x_carl[:, -1])
        error = np.linalg.norm(sol_orig.y[:, -1] - x_carl[:, -1])
        final_errors.append(error)
        print(f"Final state error (L2 norm): {error}")

    # Plot truncation order comparison
    plt.figure(figsize=(8, 6))
    plt.semilogy(truncation_orders, final_errors, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Truncation Order k')
    plt.ylabel('Final State Error (L2 norm)')
    plt.title('Carleman Approximation Error vs Truncation Order')
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"\nCarleman matrix sparsity: {np.count_nonzero(carleman_sys.Q) / carleman_sys.Q.size * 100:.1f}% non-zero")