# src/carlemann.py (Updated version with config integration)
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from itertools import combinations_with_replacement, product
from src.utils import o_plus_k_operator
import time
import logging
import os

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

    def get_system_info(self):
        """Return information about the Carleman system"""
        return {
            'original_dimension': self.n,
            'truncation_order': self.k,
            'carleman_dimension': self.Q.shape[0],
            'matrix_sparsity': np.count_nonzero(self.Q) / self.Q.size,
            'matrix_condition_number': np.linalg.cond(self.Q) if self.Q.size < 10000 else np.inf,
            'matrix_spectral_radius': np.max(np.abs(np.linalg.eigvals(self.Q))) if self.Q.size < 1000 else np.inf
        }


def simulate_and_compare(A1, A2, x0, t_span, t_eval, truncation_order=3, 
                        solver_method='Radau', rtol=1e-10, atol=1e-12, 
                        logger=None):
    """
    Simulate both original and Carleman linearized systems and compare results
    
    Parameters:
    A1, A2: System matrices
    x0: Initial conditions
    t_span: Time span tuple (t_start, t_end)
    t_eval: Time evaluation points
    truncation_order: Truncation order for Carleman approximation
    solver_method: ODE solver method
    rtol, atol: Tolerances for ODE solver
    logger: Logger instance for output
    
    Returns:
    sol_original: Original system solution
    x_carleman: Carleman approximation solution
    carleman: CarlemanLinearization instance
    """
    n_states = len(x0)

    # Create Carleman linearization Matrix
    carleman = CarlemanLinearization(A1, A2, n_states, truncation_order)

    # Initial conditions
    y0 = carleman.construct_augmented_state(x0)

    if logger:
        logger.info(f"Original system dimension: {n_states}")
        logger.info(f"Carleman system dimension: {len(y0)}")
        logger.info(f"Truncation order: {truncation_order}")

    # Solve original system
    if logger:
        logger.info("Solving original nonlinear system...")
    
    try:
        sol_original = solve_ivp(carleman.original_dynamics, t_span, x0,
                               t_eval=t_eval, method=solver_method, rtol=rtol, atol=atol)
        
        if not sol_original.success:
            if logger:
                logger.warning(f"Original system solver warning: {sol_original.message}")
    except Exception as e:
        if logger:
            logger.error(f"Original system solver failed: {str(e)}")
        raise

    # Solve Carleman linearized system
    if logger:
        logger.info("Solving Carleman linearized system...")
    
    try:
        sol_carleman = solve_ivp(carleman.carleman_dynamics, t_span, y0,
                               t_eval=t_eval, method=solver_method, rtol=rtol, atol=atol)
        
        if not sol_carleman.success:
            if logger:
                logger.warning(f"Carleman system solver warning: {sol_carleman.message}")
    except Exception as e:
        if logger:
            logger.error(f"Carleman system solver failed: {str(e)}")
        raise

    # Extract original states from Carleman solution
    x_carleman = np.array([carleman.extract_original_state(y)
                          for y in sol_carleman.y.T]).T

    if logger:
        system_info = carleman.get_system_info()
        logger.info(f"Carleman matrix sparsity: {system_info['matrix_sparsity']*100:.1f}% non-zero")
        logger.info(f"Carleman matrix condition number: {system_info['matrix_condition_number']:.2e}")

    return sol_original, x_carleman, carleman


def plot_results(t_eval, x_original, x_carleman, truncation_order, 
                system_name="Unknown System", save_path=None, show_plot=True):
    """
    Plot comparison between original and Carleman approximation
    
    Parameters:
    t_eval: Time evaluation points
    x_original: Original system solution
    x_carleman: Carleman approximation solution  
    truncation_order: Truncation order used
    system_name: Name of the system for plot title
    save_path: Path to save plot (optional)
    show_plot: Whether to display plot
    """
    n_states = x_original.shape[0]

    fig, axes = plt.subplots(2, n_states, figsize=(4*n_states, 8))
    if n_states == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(f'{system_name} - Carleman Approximation (k={truncation_order})', fontsize=14)

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
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show_plot:
        plt.show()
    else:
        plt.close()

    # Print error statistics
    print(f"\nError Statistics (Truncation order k={truncation_order}):")
    print("-" * 50)
    for i in range(n_states):
        max_error = np.max(errors[i, :])
        mean_error = np.mean(errors[i, :])
        final_error = errors[i, -1]
        print(f"State {i+1}: Max error = {max_error:.2e}, Mean error = {mean_error:.2e}, Final error = {final_error:.2e}")


def run_config_based_simulation(config_path=None, system_name=None):
    """
    Run simulation based on configuration file
    
    Parameters:
    config_path: Path to configuration file
    system_name: Name of predefined system to use
    """
    from src.config import ConfigLoader, load_system_config
    
    # Load configuration
    if system_name:
        config = load_system_config(system_name)
    else:
        config = ConfigLoader(config_path)
    
    # Setup logging
    logger = logging.getLogger('carleman_simulation')
    logger.setLevel(getattr(logging, config.logging.level))
    
    if config.logging.log_to_file:
        os.makedirs(os.path.dirname(config.logging.log_file), exist_ok=True)
        handler = logging.FileHandler(config.logging.log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Get system parameters
    A1, A2 = config.get_system_matrices()
    x0 = config.get_initial_conditions()
    t_span = config.get_time_span()
    t_eval = config.get_time_eval()
    
    logger.info(f"Starting simulation with truncation order {config.system.truncation_order}")
    
    # Run simulation
    sol_orig, x_carl, carleman_sys = simulate_and_compare(
        A1, A2, x0, t_span, t_eval, 
        truncation_order=config.system.truncation_order,
        solver_method=config.simulation.solver_method,
        rtol=config.simulation.rtol,
        atol=config.simulation.atol,
        logger=logger
    )
    
    # Plot results
    save_path = None
    if config.plotting.save_plots:
        config.create_output_directories()
        save_path = os.path.join(config.plotting.output_directory, 
                                f'simulation_k{config.system.truncation_order}.{config.plotting.save_format}')
    
    plot_results(t_eval, sol_orig.y, x_carl, config.system.truncation_order,
                system_name=system_name or "Custom System",
                save_path=save_path, show_plot=config.plotting.show_plots)
    
    return sol_orig, x_carl, carleman_sys


# Example usage and legacy compatibility
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

    plot_results(t_eval, sol_orig.y, x_carl, 4)

    # Example 2: Test different truncation orders
    print("\nExample 2: Truncation Order Comparison")
    print("=" * 40)

    truncation_orders = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    final_errors = []

    for k in truncation_orders:
        print(f"\nTesting truncation order k={k}")
        try:
            sol_orig, x_carl, _ = simulate_and_compare(
                A1, A2, x0, t_span, t_eval, truncation_order=k)
            print("Original Solution:")
            print(sol_orig.y[:, -1])
            print(sol_orig.y.shape)
            print("Carleman Solution:")
            print(x_carl[:, -1])
            print(x_carl.shape)
            error = np.linalg.norm(sol_orig.y[:, -1] - x_carl[:, -1])
            final_errors.append(error)
            print(f"Final state error (L2 norm): {error}")
        except Exception as e:
            print(f"Failed for k={k}: {str(e)}")
            final_errors.append(np.inf)

    # Plot truncation order comparison
    valid_orders = [k for k, e in zip(truncation_orders, final_errors) if np.isfinite(e)]
    valid_errors = [e for e in final_errors if np.isfinite(e)]
    
    if valid_errors:
        plt.figure(figsize=(8, 6))
        plt.semilogy(valid_orders, valid_errors, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Truncation Order k')
        plt.ylabel('Final State Error (L2 norm)')
        plt.title('Carleman Approximation Error vs Truncation Order')
        plt.grid(True, alpha=0.3)
        plt.show()

    if 'carleman_sys' in locals():
        system_info = carleman_sys.get_system_info()
        print(f"\nCarleman matrix sparsity: {system_info['matrix_sparsity']*100:.1f}% non-zero")
        print(f"Carleman matrix condition number: {system_info['matrix_condition_number']:.2e}")