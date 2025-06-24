#!/usr/bin/env python3
"""
Run Carleman Error Analysis experiments with specific configuration files.

This script allows you to run experiments on individual systems or compare multiple systems
using command line arguments and configuration files.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.experiments import ExperimentRunner
from src.config import ConfigLoader, load_system_config, list_available_systems


def run_single_system_experiment(config_path: str, truncation_orders: Optional[List[int]] = None,
                                save_plots: bool = True, show_plots: bool = False,
                                output_dir: Optional[str] = None, verbose: bool = False):
    """
    Run experiments for a single system using a config file
    
    Args:
        config_path: Path to system configuration file
        truncation_orders: List of truncation orders to test
        save_plots: Whether to save plots
        show_plots: Whether to display plots
        output_dir: Output directory override
        verbose: Enable verbose output
    """
    try:
        # Load system config
        print(f"Loading configuration from: {config_path}")
        if verbose:
            print(f"Loading configuration from: {config_path}")
        
        config = ConfigLoader(config_path)
        
        # Create runner
        print("Initializing  runner...")
        runner = ExperimentRunner(config)
        print("Done runner...")
        
        # Override truncation orders if provided
        if truncation_orders:
            runner.truncation_orders = truncation_orders
            if verbose:
                print(f"Using custom truncation orders: {truncation_orders}")
        
        # Get system parameters
        A1, A2 = config.get_system_matrices()
        x0 = config.get_initial_conditions()
        system_name = config.config['system']['name']
        
        # Override output settings if provided
        if output_dir:
            config.config['plotting']['output_directory'] = output_dir
        config.config['plotting']['save_plots'] = save_plots
        config.config['plotting']['show_plots'] = show_plots
        
        if verbose:
            print(f"System: {system_name}")
            print(f"Initial conditions: {x0}")
            print(f"Truncation orders to test: {runner.truncation_orders}")
            print(f"Output directory: {config.plotting.output_directory}")
        
        print(f"Running experiments for {system_name}...")
        
        # Run truncation order study
        results = runner.run_truncation_order_study(system_name, A1, A2, x0)
        
        # Generate plots
        print("Generating plots...")
        runner.plot_individual_system_analysis(system_name, results)
        
        # Print results summary
        print(f"\nResults Summary for {system_name}:")
        print("-" * 40)
        for result in results:
            if result.convergence:
                print(f"k={result.truncation_order:2d}: "
                      f"final_error={result.final_error:.2e}, "
                      f"time={result.computation_time:.2f}s")
            else:
                print(f"k={result.truncation_order:2d}: FAILED")
        
        print(f"\nExperiments completed for {system_name}")
        print(f"Results saved in: {config.plotting.output_directory}")
        
        return results
        
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running experiment: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_multiple_systems_comparison(config_paths: List[str], truncation_orders: Optional[List[int]] = None,
                                  save_plots: bool = True, show_plots: bool = False,
                                  output_dir: str = "results/", verbose: bool = False):
    """
    Run comparison experiments across multiple systems
    
    Args:
        config_paths: List of configuration file paths
        truncation_orders: List of truncation orders to test
        save_plots: Whether to save plots
        show_plots: Whether to display plots
        output_dir: Output directory
        verbose: Enable verbose output
    """
    try:
        # Create runner
        runner = ExperimentRunner()
        print("Initializing comparison runner...")
        # Override truncation orders if provided
        if truncation_orders:
            runner.truncation_orders = truncation_orders
        
        system_results = {}
        
        # Run experiments for each system
        for config_path in config_paths:
            if verbose:
                print(f"\nProcessing: {config_path}")
            
            config = ConfigLoader(config_path)
            A1, A2 = config.get_system_matrices()
            x0 = config.get_initial_conditions()
            system_name = config.config['system']['name']
            
            print(f"Running experiments for {system_name}...")
            results = runner.run_truncation_order_study(system_name, A1, A2, x0, config)
            system_results[system_name] = results
        
        # Generate comparison plots
        print("Generating comparison plots...")
        runner.plot_truncation_order_comparison(system_results)
        
        # Generate individual system plots
        for system_name, results in system_results.items():
            runner.plot_individual_system_analysis(system_name, results)
        
        # Generate summary report
        print("Generating summary report...")
        report = runner.generate_summary_report(system_results)
        print("\nSUMMARY REPORT:")
        print(report)
        
        print(f"\nComparison experiments completed!")
        print(f"Results saved in: {output_dir}")
        
        return system_results
        
    except Exception as e:
        print(f"Error running comparison experiments: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def list_systems():
    """List available system configurations"""
    try:
        systems = list_available_systems()
        print("Available system configurations:")
        for system in systems:
            config_path = f"config/{system}_config.yaml"
            if os.path.exists(config_path):
                try:
                    config = load_system_config(system)
                    system_name = config.config['system']['name']
                    description = config.config['system'].get('description', 'No description')
                    print(f"  {system:15} -> {config_path}")
                    print(f"                    {system_name}: {description}")
                except:
                    print(f"  {system:15} -> {config_path} (invalid config)")
        
        if not systems:
            print("No system configurations found in config/ directory")
            print("Please add configuration files like: config/vanderpol_config.yaml")
    
    except Exception as e:
        print(f"Error listing systems: {str(e)}")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Run Carleman Error Analysis experiments with specific configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single system experiment
  python run_experiment.py config/vanderpol_config.yaml
  
  # Run with custom truncation orders
  python run_experiment.py config/vanderpol_config.yaml -k 2,3,4,5
  
  # Run multiple systems comparison
  python run_experiment.py config/vanderpol_config.yaml config/duffing_config.yaml --compare
  
  # Run with custom output directory
  python run_experiment.py config/vanderpol_config.yaml -o my_results/
  
  # Show plots during execution
  python run_experiment.py config/vanderpol_config.yaml --show-plots
  
  # List available systems
  python run_experiment.py --list-systems
        """
    )
    
    # Main arguments
    parser.add_argument('config_files', nargs='*', 
                       help='Path(s) to system configuration file(s)')
    
    # Experiment options
    parser.add_argument('-k', '--truncation-orders', type=str,
                       help='Comma-separated list of truncation orders (e.g., "2,3,4,5")')
    
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                       help='Output directory for results (default: from config file)')
    
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison between multiple systems')
    
    # Plot options
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save plots to files (default: True)')
    
    parser.add_argument('--no-save-plots', dest='save_plots', action='store_false',
                       help='Do not save plots to files')
    
    parser.add_argument('--show-plots', action='store_true', default=False,
                       help='Display plots during execution')
    
    # Utility options
    parser.add_argument('--list-systems', action='store_true',
                       help='List available system configurations and exit')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle list systems option
    if args.list_systems:
        list_systems()
        return
    
    # Validate arguments
    if not args.config_files:
        parser.error("No configuration files provided. Use --list-systems to see available options.")
    
    # Check if config files exist
    for config_file in args.config_files:
        if not os.path.exists(config_file):
            print(f"Error: Configuration file not found: {config_file}")
            sys.exit(1)
    
    # Parse truncation orders
    truncation_orders = None
    if args.truncation_orders:
        try:
            truncation_orders = [int(k.strip()) for k in args.truncation_orders.split(',')]
        except ValueError:
            parser.error("Invalid truncation orders format. Use comma-separated integers (e.g., '2,3,4,5')")
    
    # Run experiments
    if len(args.config_files) == 1 and not args.compare:
        # Single system experiment
        run_single_system_experiment(
            config_path=args.config_files[0],
            truncation_orders=truncation_orders,
            save_plots=args.save_plots,
            show_plots=args.show_plots,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
    else:
        # Multiple systems comparison
        output_dir = args.output_dir or "results/"
        run_multiple_systems_comparison(
            config_paths=args.config_files,
            truncation_orders=truncation_orders,
            save_plots=args.save_plots,
            show_plots=args.show_plots,
            output_dir=output_dir,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()