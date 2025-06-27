#!/bin/bash

# experiment.sh - Comprehensive testing script for Carleman Error Analysis
# This script runs various experiments with different nonlinear differential equations

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE} $1 ${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Default parameters
PYTHON_CMD="python3"
CONFIG_FILE="config/default_config.yaml"
OUTPUT_DIR="results"
RUN_TESTS=true
RUN_EXPERIMENTS=true
CLEAN_OUTPUT=false
VERBOSE=false
SYSTEMS="all"
TRUNCATION_ORDERS="2,3,4,5,6,7,8"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help              Show this help message"
    echo "  -c, --config FILE       Use custom config file (default: $CONFIG_FILE)"
    echo "  -o, --output DIR        Output directory (default: $OUTPUT_DIR)"
    echo "  -p, --python CMD        Python command (default: $PYTHON_CMD)"
    echo "  --skip-tests            Skip running unit tests"
    echo "  --skip-experiments      Skip running experiments"
    echo "  --clean                 Clean output directory before running"
    echo "  --verbose               Enable verbose output"
    echo "  --systems LIST          Comma-separated list of systems to test"
    echo "                          Options: linear,vanderpol,duffing,lotka_volterra,brusselator,all"
    echo "  --truncation-orders LIST Comma-separated list of truncation orders"
    echo "                          (default: $TRUNCATION_ORDERS)"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                                    # Run all tests and experiments"
    echo "  $0 --skip-tests                      # Run only experiments"
    echo "  $0 --systems vanderpol,duffing       # Test only specific systems"
    echo "  $0 --truncation-orders 2,3,4         # Test only specific orders"
    echo "  $0 --clean --verbose                 # Clean output and run with verbose"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -p|--python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        --skip-tests)
            RUN_TESTS=false
            shift
            ;;
        --skip-experiments)
            RUN_EXPERIMENTS=false
            shift
            ;;
        --clean)
            CLEAN_OUTPUT=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --systems)
            SYSTEMS="$2"
            shift 2
            ;;
        --truncation-orders)
            TRUNCATION_ORDERS="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python and packages
check_dependencies() {
    print_header "Checking Dependencies"
    
    # Check Python
    if ! command_exists "$PYTHON_CMD"; then
        print_error "Python command '$PYTHON_CMD' not found"
        exit 1
    fi
    
    python_version=$($PYTHON_CMD --version 2>&1)
    print_success "Found $python_version"
    
    # Check required packages
    required_packages=("numpy" "scipy" "matplotlib" "yaml" "pytest")
    
    for package in "${required_packages[@]}"; do
        if $PYTHON_CMD -c "import $package" 2>/dev/null; then
            print_success "Package '$package' is available"
        else
            print_error "Required package '$package' is not installed"
            print_info "Install with: pip install $package"
            exit 1
        fi
    done
    
    # Check optional packages
    optional_packages=("seaborn")
    for package in "${optional_packages[@]}"; do
        if $PYTHON_CMD -c "import $package" 2>/dev/null; then
            print_success "Optional package '$package' is available"
        else
            print_warning "Optional package '$package' is not installed"
        fi
    done
}

# Function to setup directories
setup_directories() {
    print_header "Setting Up Directories"
    
    # Create necessary directories
    directories=("$OUTPUT_DIR" "logs" "config")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_success "Created directory: $dir"
        else
            print_info "Directory already exists: $dir"
        fi
    done
    
    # Clean output directory if requested
    if [ "$CLEAN_OUTPUT" = true ]; then
        print_info "Cleaning output directory: $OUTPUT_DIR"
        rm -rf "$OUTPUT_DIR"/*
        print_success "Output directory cleaned"
    fi
}

# Function to create default config if it doesn't exist
create_default_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        print_info "Creating default configuration file: $CONFIG_FILE"
        
        cat > "$CONFIG_FILE" << EOF
# Default configuration for Carleman Error Analysis
system:
  n_states: 2
  truncation_order: 4
  
simulation:
  t_start: 0.0
  t_end: 10.0
  n_points: 200
  solver_method: 'Radau'
  rtol: 1.0e-10
  atol: 1.0e-12

initial_conditions:
  x0: [0.333, 0.333]

matrices:
  A1: [[-0.000, 8.0],
       [-8.0, -0.000]]
  A2: [[0.1, 0.0, 0.0, 0.0],
       [0.0, 0.1, 0.0, 0.0]]

experiments:
  truncation_orders: [2, 3, 4, 5, 6, 7, 8]
  systems_to_test: ['vanderpol', 'duffing', 'lotka', 'linear', 'weakly']
  
plotting:
  figure_size: [12, 8]
  dpi: 300
  save_format: 'png'
  save_plots: true
  output_directory: '$OUTPUT_DIR'
  show_plots: false

logging:
  level: 'INFO'
  log_to_file: true
  log_file: 'logs/carleman_analysis.log'

performance:
  use_parallel: false
  n_workers: 4
  memory_limit_gb: 8
EOF
        print_success "Default configuration created"
    else
        print_info "Using existing configuration file: $CONFIG_FILE"
    fi
}

# Function to create base config and check system configs
create_base_config() {
    if [ ! -f "config/base_config.yaml" ]; then
        print_info "Creating base configuration template: config/base_config.yaml"
        
        mkdir -p config
        cat > "config/base_config.yaml" << 'EOF'
# config/base_config.yaml
# Base configuration template - copy and modify for specific systems

# System identification
system:
  name: "Generic System"
  description: "Template configuration for nonlinear systems"
  n_states: 2
  truncation_order: 4

# System matrices (MODIFY THESE FOR YOUR SYSTEM)
matrices:
  # Linear part: A1 * x
  # Should be n_states × n_states matrix
  A1: [[0.0, 1.0],
       [-1.0, 0.0]]
  
  # Quadratic part: A2 * (x ⊗ x)  
  # Should be n_states × (n_states²) matrix
  # Kronecker product ordering: [x1*x1, x1*x2, x2*x1, x2*x2] for 2D system
  A2: [[0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0]]

# Initial conditions
initial_conditions:
  x0: [1.0, 0.0]
  description: "Default initial condition"

# Simulation parameters  
simulation:
  t_start: 0.0
  t_end: 10.0
  n_points: 200
  solver_method: 'Radau'
  rtol: 1.0e-10
  atol: 1.0e-12

# System-specific parameters (optional)
parameters: {}

# Expected behavior (for documentation)
expected_behavior:
  type: "unknown"
  description: "Describe expected system behavior"

# Plotting parameters
plotting:
  figure_size: [12, 8]
  dpi: 300
  save_format: 'png'
  save_plots: true
  output_directory: 'results/'
  show_plots: false
  phase_plot: true

# Logging
logging:
  level: 'INFO'
  log_to_file: true
  log_file: 'logs/carleman_analysis.log'

# Performance
performance:
  use_parallel: false
  n_workers: 4
  memory_limit_gb: 8
EOF
        print_success "Base configuration template created"
    else
        print_info "Base configuration template exists: config/base_config.yaml"
    fi
    
    # Check for system-specific configurations and create if missing
    print_info "Checking system-specific configuration files..."
    
    # Create Van der Pol config if missing
    if [ ! -f "config/vanderpol_config.yaml" ]; then
        print_info "Creating Van der Pol configuration..."
        cat > "config/vanderpol_config.yaml" << 'EOF'
# config/vanderpol_config.yaml
# Van der Pol Oscillator Configuration
system:
  name: "Van der Pol Oscillator"
  description: "Classic nonlinear oscillator with self-sustaining oscillations"
  n_states: 2
  truncation_order: 4
  
matrices:
  A1: [[0.0, 1.0], [-1.0, 1.0]]
  A2: [[0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]]

initial_conditions:
  x0: [2.0, 0.0]

simulation:
  t_start: 0.0
  t_end: 20.0
  n_points: 400
  solver_method: 'Radau'
  rtol: 1.0e-10
  atol: 1.0e-12

parameters:
  mu: 1.0

plotting:
  figure_size: [12, 8]
  dpi: 300
  save_format: 'png'
  save_plots: true
  output_directory: 'results/vanderpol/'
  show_plots: false

logging:
  level: 'INFO'
  log_to_file: true
  log_file: 'logs/vanderpol_analysis.log'

performance:
  use_parallel: false
  n_workers: 4
  memory_limit_gb: 8
EOF
        print_success "Created Van der Pol configuration"
    else
        print_success "Found Van der Pol configuration"
    fi
    
    # Create Linear system config if missing
    if [ ! -f "config/linear_config.yaml" ]; then
        print_info "Creating Linear system configuration..."
        cat > "config/linear_config.yaml" << 'EOF'
# config/linear_config.yaml
# Linear System Configuration
system:
  name: "Linear System"
  description: "Stable linear system for validation"
  n_states: 2
  truncation_order: 2
  
matrices:
  A1: [[-0.1, 1.0], [-1.0, -0.1]]
  A2: [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]

initial_conditions:
  x0: [1.0, 0.0]

simulation:
  t_start: 0.0
  t_end: 20.0
  n_points: 200
  solver_method: 'RK45'
  rtol: 1.0e-8
  atol: 1.0e-10

plotting:
  figure_size: [10, 8]
  dpi: 300
  save_format: 'png'
  save_plots: true
  output_directory: 'results/linear/'
  show_plots: false

logging:
  level: 'INFO'
  log_to_file: true
  log_file: 'logs/linear_analysis.log'

performance:
  use_parallel: false
  n_workers: 2
  memory_limit_gb: 4
EOF
        print_success "Created Linear system configuration"
    else
        print_success "Found Linear system configuration"
    fi
    
    # Show available configurations
    print_info "Available system configurations:"
    for config_file in config/*_config.yaml; do
        if [ -f "$config_file" ]; then
            basename_file=$(basename "$config_file" _config.yaml)
            print_success "  - $basename_file"
        fi
    done
}

# Function to run unit tests
run_tests() {
    print_header "Running Unit Tests"
    
    if [ "$RUN_TESTS" = false ]; then
        print_warning "Skipping unit tests (--skip-tests specified)"
        return 0
    fi
    
    # Run comprehensive tests
    print_info "Running comprehensive test suite..."
    
    if [ "$VERBOSE" = true ]; then
        $PYTHON_CMD -m pytest tests/ -v --tb=short
    else
        $PYTHON_CMD tests/test_comprehensive.py
    fi
    
    if [ $? -eq 0 ]; then
        print_success "All tests passed!"
    else
        print_error "Some tests failed!"
        exit 1
    fi
}

# Function to run individual system experiment
run_system_experiment() {
    local system_name=$1
    print_info "Running experiment for $system_name system..."
    
    # No need for temporary config - we use system-specific configs directly
    
    # Run the experiment using system-specific config
    if [ "$VERBOSE" = true ]; then
        $PYTHON_CMD -c "
import sys
sys.path.append('.')
from src.config import load_system_config
from src.experiments import ExperimentRunner

try:
    # Load system-specific configuration
    config = load_system_config('$system_name')
    print("Loaded configuration for $system_name:")
    # Create runner with system config
    runner = ExperimentRunner(config)
    print("Created Experiment Runner for $system_name:")
    # Get system matrices and initial conditions
    A1, A2 = config.get_system_matrices()
    x0 = config.get_initial_conditions()
    system_display_name = config.config['system']['name']
    print("Loaded initial conditions and system matrices for $system_name:")
    # Run truncation order study
    results = runner.run_truncation_order_study(system_display_name, A1, A2, x0)
    runner.plot_individual_system_analysis(system_display_name, results)
    
    print(f'Completed experiment for {system_display_name}')
    
except Exception as e:
    print(f'Error running experiment for $system_name: {str(e)}')
    sys.exit(1)
"
    else
        $PYTHON_CMD -c "
import sys
sys.path.append('.')
from src.config import load_system_config
from src.experiments import ExperimentRunner

try:
    config = load_system_config('$system_name')
    runner = ExperimentRunner(config)
    A1, A2 = config.get_system_matrices()
    x0 = config.get_initial_conditions()
    system_display_name = config.config['system']['name']
    results = runner.run_truncation_order_study(system_display_name, A1, A2, x0)
    runner.plot_individual_system_analysis(system_display_name, results)
except Exception as e:
    print(f'Error: {str(e)}')
    sys.exit(1)
" 2>/dev/null
    fi
    
    # Clean up temporary config
    # rm -f "$temp_config"  # No longer needed since we use system configs directly
    
    if [ $? -eq 0 ]; then
        print_success "Experiment completed for $system_name"
    else
        print_error "Experiment failed for $system_name"
        return 1
    fi
}

# Function to run experiments
run_experiments() {
    print_header "Running Experiments"
    
    if [ "$RUN_EXPERIMENTS" = false ]; then
        print_warning "Skipping experiments (--skip-experiments specified)"
        return 0
    fi
    
    # Determine which systems to test
    if [ "$SYSTEMS" = "all" ]; then
        systems_to_test=("linear" "vanderpol" "duffing" "lotka_volterra" "brusselator")  # Start with basic systems
    else
        IFS=',' read -ra systems_to_test <<< "$SYSTEMS"
    fi
    
    print_info "Testing systems: ${systems_to_test[*]}"
    print_info "Truncation orders: $TRUNCATION_ORDERS"
    
    # Run individual system experiments
    failed_systems=()
    for system in "${systems_to_test[@]}"; do
        if ! run_system_experiment "$system"; then
            failed_systems+=("$system")
        fi
    done
    
    # Run comprehensive comparison
    print_info "Running comprehensive system comparison..."
    
    if [ "$VERBOSE" = true ]; then
        $PYTHON_CMD src/experiments.py
    else
        $PYTHON_CMD src/experiments.py 2>/dev/null
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Comprehensive experiment completed"
    else
        print_error "Comprehensive experiment failed"
        return 1
    fi
    
    # Report results
    if [ ${#failed_systems[@]} -eq 0 ]; then
        print_success "All system experiments completed successfully!"
    else
        print_warning "Some system experiments failed: ${failed_systems[*]}"
    fi
}

# Function to generate performance benchmark
run_performance_benchmark() {
    print_header "Running Performance Benchmark"
    
    print_info "Benchmarking different truncation orders..."
    
    # Create benchmark script
    cat > /tmp/benchmark.py << 'EOF'
import sys
sys.path.append('.')
import time
import numpy as np
from src.carlemann import CarlemanLinearization
from src.experiments import SystemLibrary

def benchmark_truncation_order(k_max=10):
    """Benchmark different truncation orders"""
    library = SystemLibrary()
    A1, A2, x0, _ = library.get_vanderpol_system()
    
    results = []
    for k in range(2, k_max + 1):
        try:
            start_time = time.time()
            carleman = CarlemanLinearization(A1, A2, 2, k)
            construction_time = time.time() - start_time
            
            matrix_size = carleman.Q.shape[0]
            memory_usage = carleman.Q.nbytes / (1024 * 1024)  # MB
            
            results.append({
                'k': k,
                'matrix_size': matrix_size,
                'construction_time': construction_time,
                'memory_usage_mb': memory_usage
            })
            
            print(f"k={k:2d}: size={matrix_size:5d}, time={construction_time:.4f}s, memory={memory_usage:.2f}MB")
            
        except Exception as e:
            print(f"k={k:2d}: FAILED - {str(e)}")
            break
    
    return results

if __name__ == "__main__":
    print("Truncation Order Benchmark:")
    print("k   | Matrix Size | Time (s) | Memory (MB)")
    print("----|-------------|----------|------------")
    benchmark_truncation_order()
EOF
    
    $PYTHON_CMD /tmp/benchmark.py
    rm -f /tmp/benchmark.py
    
    print_success "Performance benchmark completed"
}

# Function to generate final report
generate_report() {
    print_header "Generating Final Report"
    
    report_file="$OUTPUT_DIR/experiment_report.md"
    
    cat > "$report_file" << EOF
# Carleman Error Analysis - Experiment Report

**Generated on:** $(date)
**Configuration:** $CONFIG_FILE
**Systems tested:** $SYSTEMS
**Truncation orders:** $TRUNCATION_ORDERS

## Summary

This report contains the results of running the Carleman Error Analysis experiments.

### Files Generated

EOF
    
    # List generated files
    if [ -d "$OUTPUT_DIR" ]; then
        echo "### Generated Files:" >> "$report_file"
        echo "" >> "$report_file"
        find "$OUTPUT_DIR" -type f -name "*.png" -o -name "*.txt" -o -name "*.log" | sort | while read file; do
            echo "- \`$(basename "$file")\`" >> "$report_file"
        done
        echo "" >> "$report_file"
    fi
    
    # Add system information
    cat >> "$report_file" << EOF

### System Information

- **Python Version:** $($PYTHON_CMD --version)
- **NumPy Version:** $($PYTHON_CMD -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "Not available")
- **SciPy Version:** $($PYTHON_CMD -c "import scipy; print(scipy.__version__)" 2>/dev/null || echo "Not available")
- **Matplotlib Version:** $($PYTHON_CMD -c "import matplotlib; print(matplotlib.__version__)" 2>/dev/null || echo "Not available")

### Usage

To view the results:

1. **Plots**: Open the PNG files in the results directory
2. **Summary Report**: Read \`experiment_summary_report.txt\`
3. **Logs**: Check \`logs/carleman_analysis.log\` for detailed execution logs

### Recommendations

Based on the experiment results, refer to the summary report for system-specific recommendations and optimal truncation order suggestions.

EOF
    
    print_success "Report generated: $report_file"
}

# Function to show final summary
show_summary() {
    print_header "Experiment Summary"
    
    echo "Configuration used: $CONFIG_FILE"
    echo "Output directory: $OUTPUT_DIR"  
    echo "Systems tested: $SYSTEMS"
    echo "Truncation orders: $TRUNCATION_ORDERS"
    echo ""
    
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Generated files:"
        find "$OUTPUT_DIR" -type f | sort | while read file; do
            echo "  - $file"
        done
        echo ""
    fi
    
    if [ -f "logs/carleman_analysis.log" ]; then
        echo "Detailed logs available in: logs/carleman_analysis.log"
    fi
    
    print_success "All experiments completed successfully!"
    print_info "Check the results directory for plots and reports: $OUTPUT_DIR"
}

# Main execution
main() {
    print_header "Carleman Error Analysis - Experiment Runner"
    
    # Check dependencies
    check_dependencies
    
    # Setup directories and config
    setup_directories
    create_base_config
    
    # Run tests
    run_tests
    
    # Run experiments  
    run_experiments
    
    # Run performance benchmark
    run_performance_benchmark
    
    # Generate report
    generate_report
    
    # Show summary
    show_summary
}

# Run main function
main "$@"