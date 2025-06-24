# Carleman Error Analysis

A comprehensive Python package for analyzing the approximation accuracy of Carleman linearization for nonlinear dynamical systems.

## Overview

Carleman linearization is a powerful technique for converting nonlinear dynamical systems into equivalent linear systems of higher dimension. This package provides tools to:

- Construct Carleman linearizations for polynomial nonlinear systems
- Analyze approximation accuracy across different truncation orders
- Compare performance on various nonlinear dynamical systems
- Visualize error evolution and convergence properties

## Features

- **Flexible Configuration**: YAML-based configuration system for easy parameter management
- **Multiple System Types**: Built-in support for Van der Pol, Duffing, Lotka-Volterra, and other classic nonlinear systems
- **Comprehensive Testing**: Extensive unit test suite with numerical validation
- **Automated Experiments**: Shell script for running batch experiments across multiple systems
- **Rich Visualization**: Matplotlib-based plotting with customizable output formats
- **Performance Analysis**: Built-in benchmarking and computational cost analysis

## Installation

### Prerequisites

- Python 3.7+
- NumPy, SciPy, Matplotlib
- PyYAML for configuration management

### Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd carleman-error-analysis

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```python
import numpy as np
from src.carlemann import CarlemanLinearization, simulate_and_compare

# Define a simple nonlinear system: Van der Pol oscillator
A1 = np.array([[0, 1], [-1, 1]])           # Linear part
A2 = np.array([[0, 0, 0, 0], [0, -1, 0, 0]])  # Quadratic part

# Initial conditions and time span
x0 = np.array([2.0, 0.0])
t_span = (0, 10)
t_eval = np.linspace(0, 10, 200)

# Run simulation and comparison
sol_orig, x_carl, carleman_sys = simulate_and_compare(
    A1, A2, x0, t_span, t_eval, truncation_order=4
)

print("Carleman system dimension:", carleman_sys.Q.shape[0])
print("Final approximation error:", np.linalg.norm(sol_orig.y[:, -1] - x_carl[:, -1]))
```

### Configuration-Based Usage

```python
from src.config import ConfigLoader
from src.carlemann import run_config_based_simulation

# Load configuration
config = ConfigLoader('config/default_config.yaml')

# Run simulation
sol_orig, x_carl, carleman_sys = run_config_based_simulation('config/default_config.yaml')
```

### Predefined Systems

```python
from src.config import load_system_config

# Load Van der Pol system configuration
config = load_system_config('vanderpol')

# Available systems: 'vanderpol', 'duffing', 'lorenz_2d'
```

## Running Experiments

### Comprehensive Experiment Suite

Use the provided shell script to run comprehensive experiments:

```bash
# Run all experiments with default settings
./experiment.sh

# Run experiments for specific systems
./experiment.sh --systems vanderpol,duffing

# Run with custom truncation orders
./experiment.sh --truncation-orders 2,3,4,5

# Skip tests and run only experiments
./experiment.sh --skip-tests --verbose

# Clean output directory and run with custom config
./experiment.sh --clean --config my_config.yaml
```

### Available Options

- `--help`: Show detailed usage information
- `--config FILE`: Use custom configuration file
- `--output DIR`: Specify output directory
- `--systems LIST`: Test specific systems (comma-separated)
- `--truncation-orders LIST`: Test specific truncation orders
- `--skip-tests`: Skip unit tests
- `--skip-experiments`: Skip experiments (tests only)
- `--clean`: Clean output directory before running
- `--verbose`: Enable verbose output

### Python-Based Experiments

```python
from src.experiments import ExperimentRunner
from src.config import ConfigLoader

# Load configuration
config = ConfigLoader()

# Create experiment runner
runner = ExperimentRunner(config)

# Run comprehensive system comparison
system_results = runner.run_system_comparison()

# Generate plots and reports
runner.plot_truncation_order_comparison(system_results)
report = runner.generate_summary_report(system_results)
print(report)
```

## Configuration

The package uses YAML configuration files for flexible parameter management. Key configuration sections:

### System Parameters
```yaml
system:
  n_states: 2
  truncation_order: 4
```

### Simulation Settings
```yaml
simulation:
  t_start: 0.0
  t_end: 10.0
  n_points: 200
  solver_method: 'Radau'
  rtol: 1.0e-10
  atol: 1.0e-12
```

### Experiment Configuration
```yaml
experiments:
  truncation_orders: [2, 3, 4, 5, 6, 7, 8]
  systems_to_test: ['vanderpol', 'duffing', 'lotka']
```

### Plotting Options
```yaml
plotting:
  figure_size: [12, 8]
  dpi: 300
  save_format: 'png'
  save_plots: true
  output_directory: 'results/'
  show_plots: false
```

## Testing

### Run Unit Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python tests/test_comprehensive.py

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Categories

- **Kronecker Operations**: Test mathematical operations and matrix constructions
- **Carleman Linearization**: Test system construction and dynamics
- **Configuration Management**: Test YAML loading and parameter handling
- **Integration Tests**: Test complete simulation workflows
- **Numerical Stability**: Test edge cases and stability limits

## Project Structure

```
carleman-error-analysis/
├── src/
│   ├── __init__.py
│   ├── carlemann.py          # Core Carleman linearization
│   ├── utils.py              # Mathematical utilities
│   ├── config.py             # Configuration management
│   └── experiments.py        # Experiment runner
├── tests/
│   ├── __init__.py
│   ├── test.py               # Basic tests
│   └── test_comprehensive.py # Comprehensive test suite
├── config/
│   └── default_config.yaml   # Default configuration
├── results/                  # Output directory (created automatically)
├── logs/                     # Log files (created automatically)
├── experiment.sh             # Experiment runner script
├── setup.py                  # Package setup
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Built-in Systems

The package includes several predefined nonlinear systems:

### Van der Pol Oscillator
```
dx₁/dt = x₂
dx₂/dt = μ(1 - x₁²)x₂ - x₁
```

### Duffing Oscillator
```
dx₁/dt = x₂
dx₂/dt = -γx₂ + αx₁ + βx₁³
```

### Lotka-Volterra System
```
dx₁/dt = ax₁ - bx₁x₂
dx₂/dt = -cx₂ + dx₁x₂
```

### Weakly Nonlinear System
```
dx₁/dt = -x₁ + x₂ + εx₁x₂
dx₂/dt = -x₁ - x₂ + εx₁²
```

## Output and Results

### Generated Files

- **Plots**: PNG/PDF files showing trajectory comparisons and error analysis
- **Reports**: Text-based summary reports with statistics and recommendations
- **Logs**: Detailed execution logs for debugging and analysis
- **Data**: CSV files with numerical results (optional)

### Key Metrics

- **Final Error**: L2 norm of state difference at final time
- **Max Error**: Maximum error over entire simulation
- **Mean Error**: Average error over time
- **Computation Time**: Wall-clock time for Carleman matrix construction and simulation
- **Convergence**: Success/failure status of numerical integration

## Mathematical Background

### Carleman Linearization Theory

For a polynomial nonlinear system:
```
dx/dt = A₁x + A₂(x⊗x) + A₃(x⊗x⊗x) + ...
```

The Carleman linearization constructs an equivalent linear system:
```
dy/dt = Qy
```

where `y = [x, x⊗x, x⊗x⊗x, ...]` is the augmented state vector and `Q` is the Carleman matrix.

### Truncation and Approximation

The infinite-dimensional system is truncated at order `k`, yielding a finite-dimensional approximation. The package analyzes how approximation quality varies with truncation order.

## Performance Considerations

### Computational Complexity

- **Matrix Construction**: O(n^(2k)) for truncation order k and state dimension n
- **Memory Usage**: O(n^(2k)) storage for Carleman matrix
- **Simulation**: Standard ODE solver complexity on enlarged system

### Recommended Usage

- Use truncation orders 2-8 for most applications
- Monitor memory usage for high-dimensional systems
- Consider parallel processing for parameter studies

## Contributing

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 mypy

# Run code formatting
black src/ tests/

# Run type checking
mypy src/

# Run linting
flake8 src/ tests/
```

### Adding New Systems

1. Add system definition to `SystemLibrary` class in `experiments.py`
2. Add configuration entry to `SYSTEM_CONFIGS` in `config.py`
3. Update tests to include new system
4. Document the system in README

### Testing Guidelines

- Add unit tests for new mathematical operations
- Include integration tests for new systems
- Test edge cases and numerical stability
- Maintain >90% code coverage

## Troubleshooting

### Common Issues

**Memory Errors**: Reduce truncation order or system dimension

**Convergence Failures**: Adjust ODE solver tolerances or try different solvers

**Configuration Errors**: Validate YAML syntax and parameter ranges

**Import Errors**: Ensure all dependencies are installed with correct versions

### Debug Mode

Enable verbose logging:
```yaml
logging:
  level: 'DEBUG'
  log_to_file: true
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in academic work, please cite:

```bibtex
@software{carleman_error_analysis,
  title={Carleman Error Analysis: A Python Package for Nonlinear System Approximation},
  author={Abhinav Muraleedharan},
  year={2025},
  url={https://github.com/Abhinav-Muraleedharan/carleman-error-analysis}
}
```

## Contact

For questions, suggestions, or bug reports, please open an issue on GitHub or contact [abhi@cs.toronto.edu].

---

**Note**: This package is designed for research and educational purposes. For production applications, consider additional validation and testing for your specific use cases.
