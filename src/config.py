# src/config.py
import yaml
import os
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class SystemConfig:
    """Configuration for system parameters"""
    n_states: int
    truncation_order: int


@dataclass 
class SimulationConfig:
    """Configuration for simulation parameters"""
    t_start: float
    t_end: float
    n_points: int
    solver_method: str
    rtol: float
    atol: float


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    truncation_orders: List[int]
    systems_to_test: List[str]


@dataclass
class PlottingConfig:
    """Configuration for plotting"""
    figure_size: List[int]
    dpi: int
    save_format: str
    save_plots: bool
    output_directory: str
    show_plots: bool


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str
    log_to_file: bool
    log_file: str


@dataclass
class PerformanceConfig:
    """Configuration for performance settings"""
    use_parallel: bool
    n_workers: int
    memory_limit_gb: int


class ConfigLoader:
    """Load and manage configuration settings for Carleman analysis"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'default_config.yaml')
        
        self.config_path = config_path
        self.config = self._load_config()
        
        # Parse configurations into structured objects
        self.system = SystemConfig(**self.config['system'])
        self.simulation = SimulationConfig(**self.config['simulation'])
        self.experiments = ExperimentConfig(**self.config['experiments'])
        self.plotting = PlottingConfig(**self.config['plotting'])
        self.logging = LoggingConfig(**self.config['logging'])
        self.performance = PerformanceConfig(**self.config['performance'])
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
    
    def get_initial_conditions(self) -> np.ndarray:
        """Get initial conditions as numpy array"""
        return np.array(self.config['initial_conditions']['x0'])
    
    def get_system_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Get system matrices A1 and A2"""
        A1 = np.array(self.config['matrices']['A1'])
        A2 = np.array(self.config['matrices']['A2'])
        return A1, A2
    
    def get_time_span(self) -> tuple[float, float]:
        """Get time span for simulation"""
        return (self.simulation.t_start, self.simulation.t_end)
    
    def get_time_eval(self) -> np.ndarray:
        """Get time evaluation points"""
        return np.linspace(self.simulation.t_start, self.simulation.t_end, self.simulation.n_points)
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        self._deep_update(self.config, updates)
        
        # Re-parse configurations
        self.system = SystemConfig(**self.config['system'])
        self.simulation = SimulationConfig(**self.config['simulation'])
        self.experiments = ExperimentConfig(**self.config['experiments'])
        self.plotting = PlottingConfig(**self.config['plotting'])
        self.logging = LoggingConfig(**self.config['logging'])
        self.performance = PerformanceConfig(**self.config['performance'])
    
    def _deep_update(self, original: Dict, updates: Dict):
        """Recursively update nested dictionary"""
        for key, value in updates.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value
    
    def save_config(self, output_path: str):
        """Save current configuration to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
    
    def create_output_directories(self):
        """Create necessary output directories"""
        directories = [
            self.plotting.output_directory,
            os.path.dirname(self.logging.log_file) if self.logging.log_to_file else None
        ]
        
        for directory in directories:
            if directory:
                os.makedirs(directory, exist_ok=True)


# Available system configuration files
AVAILABLE_SYSTEMS = {
    'vanderpol': 'config/vanderpol_config.yaml',
    'duffing': 'config/duffing_config.yaml',
    'lotka_volterra': 'config/lotka_volterra_config.yaml',
    'linear': 'config/linear_system_config.yaml',
    'brusselator': 'config/brusselator_config.yaml'
}


def load_system_config(system_name: str) -> ConfigLoader:
    """
    Load configuration for a specific predefined system
    
    Args:
        system_name: Name of the predefined system
        
    Returns:
        ConfigLoader instance with system-specific configuration
    """
    if system_name not in AVAILABLE_SYSTEMS:
        raise ValueError(f"Unknown system: {system_name}. Available systems: {list(AVAILABLE_SYSTEMS.keys())}")
    
    config_path = AVAILABLE_SYSTEMS[system_name]
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return ConfigLoader(config_path)


def list_available_systems() -> List[str]:
    """Return list of available predefined systems"""
    return list(AVAILABLE_SYSTEMS.keys())


def get_system_config_path(system_name: str) -> str:
    """Get the configuration file path for a system"""
    if system_name not in AVAILABLE_SYSTEMS:
        raise ValueError(f"Unknown system: {system_name}")
    return AVAILABLE_SYSTEMS[system_name]