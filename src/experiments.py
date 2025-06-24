# src/experiments.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
import os
import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.carlemann import CarlemanLinearization, simulate_and_compare
from src.config import ConfigLoader, load_system_config


@dataclass
class ExperimentResult:
    """Container for experiment results"""
    system_name: str
    truncation_order: int
    final_error: float
    max_error: float
    mean_error: float
    computation_time: float
    convergence: bool
    error_time_series: np.ndarray
    time_points: np.ndarray


class SystemLibrary:
    """Library of nonlinear dynamical systems for testing"""
    
    @staticmethod
    def get_vanderpol_system(mu: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Van der Pol oscillator: 
        dx1/dt = x2
        dx2/dt = μ(1 - x1²)x2 - x1
        """
        A1 = np.array([[0.0, 1.0], [-1.0, mu]])
        A2 = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, -mu, 0.0, 0.0]])
        x0 = np.array([2.0, 0.0])
        name = f"Van der Pol (μ={mu})"
        return A1, A2, x0, name
    
    @staticmethod
    def get_duffing_system(alpha: float = -1.0, beta: float = 1.0, gamma: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Duffing oscillator:
        dx1/dt = x2  
        dx2/dt = -γx2 + αx1 + βx1³
        """
        A1 = np.array([[0.0, 1.0], [alpha, -gamma]])
        # For x1³ term, we need x1*(x1⊗x1) which involves higher-order terms
        # Simplified to quadratic approximation: x1³ ≈ 0 for small x1
        A2 = np.array([[0.0, 0.0, 0.0, 0.0], [beta, 0.0, 0.0, 0.0]])
        x0 = np.array([1.0, 0.0])
        name = f"Duffing (α={alpha}, β={beta}, γ={gamma})"
        return A1, A2, x0, name
    
    @staticmethod
    def get_lotka_volterra_system(a: float = 1.0, b: float = 1.5, c: float = 0.75, d: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Lotka-Volterra system:
        dx1/dt = ax1 - bx1x2
        dx2/dt = -cx2 + dx1x2
        """
        A1 = np.array([[a, 0.0], [0.0, -c]])
        # x1x2 terms: [x1x1, x1x2, x2x1, x2x2] -> we want x1x2 coefficient
        A2 = np.array([[-b, 0.0, 0.0, 0.0], [0.0, d, 0.0, 0.0]])
        x0 = np.array([1.0, 1.0])
        name = f"Lotka-Volterra (a={a}, b={b}, c={c}, d={d})"
        return A1, A2, x0, name
    
    @staticmethod
    def get_brusselator_system(a: float = 1.0, b: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Brusselator system:
        dx1/dt = a - (b+1)x1 + x1²x2
        dx2/dt = bx1 - x1²x2
        
        Note: x1²x2 is a cubic term, approximated here
        """
        A1 = np.array([[a - (b+1), 0.0], [b, 0.0]])
        # Quadratic approximation (ignoring cubic terms for now)
        A2 = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        x0 = np.array([1.0, 1.0])
        name = f"Brusselator (a={a}, b={b})"
        return A1, A2, x0, name
    
    @staticmethod
    def get_linear_system() -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Simple linear system for validation:
        dx1/dt = -0.1*x1 + x2
        dx2/dt = -x1 - 0.1*x2
        """
        A1 = np.array([[-0.1, 1.0], [-1.0, -0.1]])
        A2 = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        x0 = np.array([1.0, 0.0])
        name = "Linear System"
        return A1, A2, x0, name
    
    @staticmethod
    def get_weakly_nonlinear_system(epsilon: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Weakly nonlinear system:
        dx1/dt = -x1 + x2 + ε*x1*x2
        dx2/dt = -x1 - x2 + ε*x1²
        """
        A1 = np.array([[-1.0, 1.0], [-1.0, -1.0]])
        A2 = np.array([[0.0, epsilon, 0.0, 0.0], [epsilon, 0.0, 0.0, 0.0]])
        x0 = np.array([0.5, 0.5])
        name = f"Weakly Nonlinear (ε={epsilon})"
        return A1, A2, x0, name


class ExperimentRunner:
    """Main experiment runner for Carleman analysis"""
    
    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize experiment runner
        
        Args:
            config_loader: Configuration loader instance
        """
        self.config = config_loader
        self.results = []
        self.logger = self._setup_logging()
        
        # Create output directories
        self.config.create_output_directories()
        
        # System library
        self.system_library = SystemLibrary()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('carleman_experiments')
        logger.setLevel(getattr(logging, self.config.logging.level))
        
        if self.config.logging.log_to_file:
            handler = logging.FileHandler(self.config.logging.log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run_single_experiment(self, system_name: str, A1: np.ndarray, A2: np.ndarray, 
                            x0: np.ndarray, truncation_order: int) -> ExperimentResult:
        """
        Run a single experiment with given parameters
        
        Args:
            system_name: Name of the system
            A1, A2: System matrices
            x0: Initial conditions
            truncation_order: Truncation order for Carleman approximation
            
        Returns:
            ExperimentResult object
        """
        self.logger.info(f"Running experiment: {system_name}, k={truncation_order}")
        
        start_time = time.time()
        
        try:
            t_span = self.config.get_time_span()
            t_eval = self.config.get_time_eval()
            print(f"Time span: {t_span}, Evaluation points: {t_eval}")
            # Run simulation
            sol_orig, x_carl, _ = simulate_and_compare(
                A1, A2, x0, t_span, t_eval, truncation_order=truncation_order)
            print(f"Simulation done for {system_name} with k={truncation_order}")
            computation_time = time.time() - start_time
            print("Original Solution:",sol_orig.y.shape)
            print("Carlemann Approximation:",x_carl.shape)
            # Compute errors
            errors = np.abs(sol_orig.y - x_carl)
            if errors.size == 0:
                raise ValueError("No errors computed, check simulation output.")
            error_norms = np.linalg.norm(errors, axis=0)
            
            max_error = np.max(error_norms)
            mean_error = np.mean(error_norms)
            final_error = error_norms[-1]
            
            convergence = sol_orig.success and np.all(np.isfinite(x_carl))
            print(f"Convergence: {convergence}, Final error: {final_error:.2e}")
            
            result = ExperimentResult(
                system_name=system_name,
                truncation_order=truncation_order,
                final_error=final_error,
                max_error=max_error,
                mean_error=mean_error,
                computation_time=computation_time,
                convergence=convergence,
                error_time_series=error_norms,
                time_points=t_eval
            )
            
            self.logger.info(f"Completed: {system_name}, k={truncation_order}, "
                           f"final_error={final_error:.2e}, time={computation_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {system_name}, k={truncation_order}, error: {str(e)}")
            
            # Return failed result
            return ExperimentResult(
                system_name=system_name,
                truncation_order=truncation_order,
                final_error=np.inf,
                max_error=np.inf,
                mean_error=np.inf,
                computation_time=time.time() - start_time,
                convergence=False,
                error_time_series=np.array([]),
                time_points=np.array([])
            )
    
    def run_truncation_order_study(self, system_name: str, A1: np.ndarray, A2: np.ndarray, 
                                 x0: np.ndarray) -> List[ExperimentResult]:
        """
        Run experiments with different truncation orders for a single system
        
        Args:
            system_name: Name of the system
            A1, A2: System matrices  
            x0: Initial conditions
            
        Returns:
            List of ExperimentResult objects
        """
        results = []
        
        for k in range(self.config.system.truncation_order):
            result = self.run_single_experiment(system_name, A1, A2, x0, k)
            print(f"Result for {system_name} with k={k}: Final error = {result.final_error:.2e}")
            results.append(result)
            self.results.append(result)
        
        return results
    
    def run_system_comparison(self) -> Dict[str, List[ExperimentResult]]:
        """
        Run experiments on multiple systems using their individual config files
        
        Returns:
            Dictionary mapping system names to their results
        """
        from src.config import list_available_systems, load_system_config
        
        system_results = {}
        available_systems = list_available_systems()
        
        self.logger.info(f"Available systems: {available_systems}")
        
        for system_name in available_systems:
            try:
                self.logger.info(f"Loading configuration for system: {system_name}")
                
                # Load system-specific configuration
                system_config = load_system_config(system_name)
                
                # Get system matrices and initial conditions from config
                A1, A2 = system_config.get_system_matrices()
                x0 = system_config.get_initial_conditions()
                
                # Update our experiment runner's config with system-specific settings
                self.config.update_config({
                    'simulation': {
                        't_start': system_config.simulation.t_start,
                        't_end': system_config.simulation.t_end,
                        'n_points': system_config.simulation.n_points,
                        'solver_method': system_config.simulation.solver_method,
                        'rtol': system_config.simulation.rtol,
                        'atol': system_config.simulation.atol
                    }
                })
                
                self.logger.info(f"Testing system: {system_config.config['system']['name']}")
                results = self.run_truncation_order_study(
                    system_config.config['system']['name'], A1, A2, x0)
                system_results[system_name] = results
                
            except Exception as e:
                self.logger.error(f"Failed to process system {system_name}: {str(e)}")
                continue
        
        return system_results
    
    def plot_truncation_order_comparison(self, system_results: Dict[str, List[ExperimentResult]]):
        """Plot comparison of truncation orders across systems"""
        fig, axes = plt.subplots(2, 2, figsize=self.config.plotting.figure_size)
        fig.suptitle('Carleman Approximation: Truncation Order Analysis', fontsize=16)
        
        # Plot 1: Final error vs truncation order
        ax1 = axes[0, 0]
        for system_name, results in system_results.items():
            k_values = [r.truncation_order for r in results if r.convergence]
            final_errors = [r.final_error for r in results if r.convergence]
            if k_values:
                ax1.semilogy(k_values, final_errors, 'o-', label=system_name, linewidth=2, markersize=6)
        
        ax1.set_xlabel('Truncation Order (k)')
        ax1.set_ylabel('Final Error (L2 norm)')
        ax1.set_title('Final Error vs Truncation Order')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Computation time vs truncation order
        ax2 = axes[0, 1]
        for system_name, results in system_results.items():
            k_values = [r.truncation_order for r in results if r.convergence]
            comp_times = [r.computation_time for r in results if r.convergence]
            if k_values:
                ax2.semilogy(k_values, comp_times, 's-', label=system_name, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Truncation Order (k)')
        ax2.set_ylabel('Computation Time (s)')
        ax2.set_title('Computation Time vs Truncation Order')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Error evolution over time (for a specific truncation order)
        ax3 = axes[1, 0]
        k_target = 4  # Show results for k=4
        for system_name, results in system_results.items():
            target_result = next((r for r in results if r.truncation_order == k_target and r.convergence), None)
            if target_result and len(target_result.error_time_series) > 0:
                ax3.semilogy(target_result.time_points, target_result.error_time_series, 
                           label=system_name, linewidth=2)
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Error (L2 norm)')
        ax3.set_title(f'Error Evolution Over Time (k={k_target})')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Convergence success rate
        ax4 = axes[1, 1]
        system_names = []
        success_rates = []
        
        for system_name, results in system_results.items():
            total_experiments = len(results)
            successful_experiments = sum(1 for r in results if r.convergence)
            success_rate = successful_experiments / total_experiments if total_experiments > 0 else 0
            
            system_names.append(system_name)
            success_rates.append(success_rate)
        
        bars = ax4.bar(system_names, success_rates, alpha=0.7)
        ax4.set_ylabel('Success Rate')
        ax4.set_title('Convergence Success Rate by System')
        ax4.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        if self.config.plotting.save_plots:
            filename = os.path.join(self.config.plotting.output_directory, 
                                  f'truncation_order_comparison.{self.config.plotting.save_format}')
            plt.savefig(filename, dpi=self.config.plotting.dpi, bbox_inches='tight')
            self.logger.info(f"Saved plot: {filename}")
        
        if self.config.plotting.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_individual_system_analysis(self, system_name: str, results: List[ExperimentResult]):
        """Plot detailed analysis for a single system"""
        fig, axes = plt.subplots(2, 2, figsize=self.config.plotting.figure_size)
        fig.suptitle(f'Detailed Analysis: {system_name}', fontsize=16)
        
        # Filter successful results
        successful_results = [r for r in results if r.convergence]
        
        if not successful_results:
            self.logger.warning(f"No successful results for {system_name}")
            plt.close()
            return
        
        # Plot 1: Error metrics vs truncation order
        ax1 = axes[0, 0]
        k_values = [r.truncation_order for r in successful_results]
        final_errors = [r.final_error for r in successful_results]
        max_errors = [r.max_error for r in successful_results]
        mean_errors = [r.mean_error for r in successful_results]
        
        ax1.semilogy(k_values, final_errors, 'o-', label='Final Error', linewidth=2, markersize=6)
        ax1.semilogy(k_values, max_errors, 's-', label='Max Error', linewidth=2, markersize=6)
        ax1.semilogy(k_values, mean_errors, '^-', label='Mean Error', linewidth=2, markersize=6)
        
        ax1.set_xlabel('Truncation Order (k)')
        ax1.set_ylabel('Error')
        ax1.set_title('Error Metrics vs Truncation Order')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Computation time scaling
        ax2 = axes[0, 1]
        comp_times = [r.computation_time for r in successful_results]
        ax2.semilogy(k_values, comp_times, 'o-', color='red', linewidth=2, markersize=6)
        ax2.set_xlabel('Truncation Order (k)')
        ax2.set_ylabel('Computation Time (s)')
        ax2.set_title('Computational Cost Scaling')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Error time series for different truncation orders
        ax3 = axes[1, 0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(successful_results)))
        
        for i, (result, color) in enumerate(zip(successful_results, colors)):
            if len(result.error_time_series) > 0:
                ax3.semilogy(result.time_points, result.error_time_series, 
                           label=f'k={result.truncation_order}', color=color, linewidth=2)
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Error (L2 norm)')
        ax3.set_title('Error Evolution Over Time')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Efficiency analysis (error vs computational cost)
        ax4 = axes[1, 1]
        scatter = ax4.scatter(comp_times, final_errors, c=k_values, s=100, alpha=0.7, cmap='viridis')
        ax4.set_xlabel('Computation Time (s)')
        ax4.set_ylabel('Final Error')
        ax4.set_yscale('log')
        ax4.set_title('Efficiency: Error vs Computational Cost')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Truncation Order (k)')
        
        # Annotate points
        for result in successful_results:
            ax4.annotate(f'k={result.truncation_order}', 
                        (result.computation_time, result.final_error),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        if self.config.plotting.save_plots:
            safe_name = system_name.replace(' ', '_').replace('(', '').replace(')', '')
            filename = os.path.join(self.config.plotting.output_directory, 
                                  f'{safe_name}_analysis.{self.config.plotting.save_format}')
            plt.savefig(filename, dpi=self.config.plotting.dpi, bbox_inches='tight')
            self.logger.info(f"Saved plot: {filename}")
        
        if self.config.plotting.show_plots:
            plt.show()
        else:
            plt.close()
    
    def generate_summary_report(self, system_results: Dict[str, List[ExperimentResult]]) -> str:
        """Generate a summary report of all experiments"""
        report = []
        report.append("=" * 80)
        report.append("CARLEMAN ERROR ANALYSIS - EXPERIMENT SUMMARY REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Configuration: {self.config.config_path}")
        report.append("")
        
        # Overall statistics
        total_experiments = sum(len(results) for results in system_results.values())
        successful_experiments = sum(sum(1 for r in results if r.convergence) 
                                   for results in system_results.values())
        
        report.append("OVERALL STATISTICS:")
        report.append(f"  Total experiments: {total_experiments}")
        report.append(f"  Successful experiments: {successful_experiments}")
        report.append(f"  Success rate: {successful_experiments/total_experiments:.1%}")
        report.append("")
        
        # System-by-system analysis
        for system_name, results in system_results.items():
            report.append(f"SYSTEM: {system_name}")
            report.append("-" * 40)
            
            successful_results = [r for r in results if r.convergence]
            
            if successful_results:
                best_result = min(successful_results, key=lambda r: r.final_error)
                worst_result = max(successful_results, key=lambda r: r.final_error)
                
                report.append(f"  Experiments: {len(results)} total, {len(successful_results)} successful")
                report.append(f"  Best performance: k={best_result.truncation_order}, "
                            f"final_error={best_result.final_error:.2e}")
                report.append(f"  Worst performance: k={worst_result.truncation_order}, "
                            f"final_error={worst_result.final_error:.2e}")
                
                # Convergence analysis
                final_errors = [r.final_error for r in successful_results]
                mean_final_error = np.mean(final_errors)
                std_final_error = np.std(final_errors)
                
                report.append(f"  Final error statistics: mean={mean_final_error:.2e}, "
                            f"std={std_final_error:.2e}")
                
                # Computational cost
                comp_times = [r.computation_time for r in successful_results]
                mean_comp_time = np.mean(comp_times)
                
                report.append(f"  Average computation time: {mean_comp_time:.3f}s")
            else:
                report.append(f"  No successful experiments for this system")
            
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)
        
        # Find best overall system
        if system_results:
            best_system = None
            best_error = np.inf
            
            for system_name, results in system_results.items():
                successful_results = [r for r in results if r.convergence]
                if successful_results:
                    min_error = min(r.final_error for r in successful_results)
                    if min_error < best_error:
                        best_error = min_error
                        best_system = system_name
            
            if best_system:
                report.append(f"  Best performing system: {best_system} "
                            f"(min error: {best_error:.2e})")
            
            # Truncation order recommendations
            k_performance = {}
            for results in system_results.values():
                for result in results:
                    if result.convergence:
                        k = result.truncation_order
                        if k not in k_performance:
                            k_performance[k] = []
                        k_performance[k].append(result.final_error)
            
            if k_performance:
                avg_errors = {k: np.mean(errors) for k, errors in k_performance.items()}
                best_k = min(avg_errors.keys(), key=lambda k: avg_errors[k])
                report.append(f"  Recommended truncation order: k={best_k} "
                            f"(avg error: {avg_errors[best_k]:.2e})")
        
        report_text = "\n".join(report)
        
        # Save report
        if self.config.plotting.save_plots:
            report_filename = os.path.join(self.config.plotting.output_directory, 
                                         'experiment_summary_report.txt')
            with open(report_filename, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Saved report: {report_filename}")
        
        return report_text


def main():
    """Main experiment runner function"""
    # Load configuration
    config_loader = ConfigLoader()
    
    # Create experiment runner
    runner = ExperimentRunner(config_loader)
    
    print("Starting Carleman Error Analysis Experiments")
    print("=" * 50)
    
    # Run system comparison
    system_results = runner.run_system_comparison()
    
    # Generate plots
    print("Generating comparison plots...")
    runner.plot_truncation_order_comparison(system_results)
    
    # Generate individual system plots
    for system_name, results in system_results.items():
        print(f"Generating detailed analysis for {system_name}...")
        runner.plot_individual_system_analysis(system_name, results)
    
    # Generate summary report
    print("Generating summary report...")
    report = runner.generate_summary_report(system_results)
    print("\nSUMMARY REPORT:")
    print(report)
    
    print("\nExperiments completed successfully!")
    print(f"Results saved in: {config_loader.plotting.output_directory}")


if __name__ == "__main__":
    main()