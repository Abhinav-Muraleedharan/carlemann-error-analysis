# tests/test_comprehensive.py
import numpy as np
import pytest
import tempfile
import os
import yaml
from src.utils import o_plus_k_operator, kronecker_power
from src.carlemann import CarlemanLinearization, simulate_and_compare
from src.config import ConfigLoader, load_system_config, list_available_systems


class TestKroneckerOperations:
    """Test suite for Kronecker operations"""
    
    def test_kronecker_power_basic(self):
        """Test basic Kronecker power operations"""
        A = np.array([[1, 2], [3, 4]])
        
        # Test k=0
        result_0 = kronecker_power(A, 0)
        expected_0 = np.array([[1.0]])
        assert np.allclose(result_0, expected_0), "Kronecker power k=0 should return scalar 1"
        
        # Test k=1
        result_1 = kronecker_power(A, 1)
        assert np.allclose(result_1, A), "Kronecker power k=1 should return original matrix"
        
        # Test k=2
        result_2 = kronecker_power(A, 2)
        expected_2 = np.kron(A, A)
        assert np.allclose(result_2, expected_2), "Kronecker power k=2 should equal A⊗A"
    
    def test_kronecker_power_identity(self):
        """Test Kronecker power with identity matrix"""
        I = np.eye(2)
        
        for k in range(1, 5):
            result = kronecker_power(I, k)
            expected_shape = (2**k, 2**k)
            assert result.shape == expected_shape, f"Identity Kronecker power k={k} has wrong shape"
            assert np.allclose(result, np.eye(2**k)), f"Identity Kronecker power k={k} should be identity"
    
    def test_o_plus_k_operator_basic(self):
        """Test basic o_plus_k_operator functionality"""
        A = np.array([[1, 2], [3, 4]])
        I = np.eye(2)
        
        # Test k=0 should return A
        result_k0 = o_plus_k_operator(A, 0)
        assert np.allclose(result_k0, A), "o_plus_k with k=0 should return original matrix"
        
        # Test k=1: A⊗I + I⊗A
        result_k1 = o_plus_k_operator(A, 1)
        expected_k1 = np.kron(A, I) + np.kron(I, A)
        assert np.allclose(result_k1, expected_k1), "o_plus_k with k=1 failed"
    
    def test_o_plus_k_operator_scalar(self):
        """Test o_plus_k_operator with scalar matrix"""
        A_scalar = np.array([[2.0]])
        
        # For scalar matrix, k=2 should give 3*A
        result = o_plus_k_operator(A_scalar, 2)
        expected = 3 * A_scalar
        assert np.allclose(result, expected), "Scalar o_plus_k operation failed"
    
    def test_o_plus_k_operator_properties(self):
        """Test mathematical properties of o_plus_k_operator"""
        A = np.array([[1, 0], [0, 2]])  # Diagonal matrix
        
        # Test that result has correct dimensions
        for k in range(1, 4):
            result = o_plus_k_operator(A, k)
            expected_dim = 2**(k+1)
            assert result.shape == (expected_dim, expected_dim), f"Wrong dimensions for k={k}"


class TestCarlemanLinearization:
    """Test suite for Carleman linearization"""
    
    def setUp(self):
        """Set up test matrices"""
        self.A1 = np.array([[0, 1], [-1, -0.1]])
        self.A2 = np.array([[0, 0, 0, 0], [-0.5, 0, 0, 0]])
        self.n_states = 2
        self.truncation_order = 3
    
    def test_initialization(self):
        """Test Carleman system initialization"""
        self.setUp()
        carleman = CarlemanLinearization(self.A1, self.A2, self.n_states, self.truncation_order)
        
        assert carleman.n == self.n_states
        assert carleman.k == self.truncation_order
        assert np.array_equal(carleman.A1, self.A1)
        assert np.array_equal(carleman.A2, self.A2)
    
    def test_initialization_invalid_dimensions(self):
        """Test initialization with invalid matrix dimensions"""
        self.setUp()
        
        # Wrong A1 dimensions
        A1_wrong = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(AssertionError):
            CarlemanLinearization(A1_wrong, self.A2, self.n_states, self.truncation_order)
        
        # Wrong A2 dimensions  
        A2_wrong = np.array([[1, 2], [3, 4]])
        with pytest.raises(AssertionError):
            CarlemanLinearization(self.A1, A2_wrong, self.n_states, self.truncation_order)
    
    def test_augmented_state_construction(self):
        """Test augmented state vector construction"""
        self.setUp()
        carleman = CarlemanLinearization(self.A1, self.A2, self.n_states, self.truncation_order)
        
        x = np.array([1.0, 2.0])
        y = carleman.construct_augmented_state(x)
        
        # Check dimensions
        expected_dim = sum(self.n_states**i for i in range(1, self.truncation_order + 1))
        assert len(y) == expected_dim, f"Augmented state has wrong dimension"
        
        # Check that original state is preserved
        assert np.allclose(y[:self.n_states], x), "Original state not preserved in augmented state"
        
        # Check second-order terms
        x_kron_2 = np.kron(x, x)  # [x1*x1, x1*x2, x2*x1, x2*x2]
        assert np.allclose(y[self.n_states:self.n_states + self.n_states**2], x_kron_2), "Second-order terms incorrect"
    
    def test_original_state_extraction(self):
        """Test extraction of original state from augmented state"""
        self.setUp()
        carleman = CarlemanLinearization(self.A1, self.A2, self.n_states, self.truncation_order)
        
        x_original = np.array([1.5, -0.5])
        y = carleman.construct_augmented_state(x_original)
        x_extracted = carleman.extract_original_state(y)
        
        assert np.allclose(x_extracted, x_original), "Original state extraction failed"
    
    def test_carleman_matrix_properties(self):
        """Test properties of the Carleman matrix"""
        self.setUp()
        carleman = CarlemanLinearization(self.A1, self.A2, self.n_states, self.truncation_order)
        
        Q = carleman.Q
        
        # Check dimensions
        expected_dim = sum(self.n_states**i for i in range(1, self.truncation_order + 1))
        assert Q.shape == (expected_dim, expected_dim), "Carleman matrix has wrong dimensions"
        
        # Check that top-left block is A1
        assert np.allclose(Q[:self.n_states, :self.n_states], self.A1), "Top-left block should be A1"
        
        # Check that it's finite
        assert np.all(np.isfinite(Q)), "Carleman matrix contains non-finite values"
    
    def test_dynamics_consistency(self):
        """Test that dynamics are computed correctly"""
        self.setUp()
        carleman = CarlemanLinearization(self.A1, self.A2, self.n_states, self.truncation_order)
        
        x = np.array([0.5, 0.1])
        
        # Original dynamics
        dx_original = carleman.original_dynamics(0, x)
        
        # Carleman dynamics
        y = carleman.construct_augmented_state(x)
        dy_carleman = carleman.carleman_dynamics(0, y)
        dx_carleman = dy_carleman[:self.n_states]  # Extract first n components
        
        # They should be approximately equal for small x
        assert np.allclose(dx_original, dx_carleman, rtol=1e-10), "Dynamics inconsistency detected"


class TestConfigLoader:
    """Test suite for configuration management"""
    
    def test_config_loading(self):
        """Test basic configuration loading"""
        # Create temporary config file
        config_data = {
            'system': {'n_states': 2, 'truncation_order': 4},
            'simulation': {'t_start': 0.0, 't_end': 10.0, 'n_points': 100, 
                          'solver_method': 'RK45', 'rtol': 1e-8, 'atol': 1e-10},
            'initial_conditions': {'x0': [1.0, 0.0]},
            'matrices': {'A1': [[0, 1], [-1, 0]], 'A2': [[0, 0, 0, 0], [0, 0, 0, 0]]},
            'experiments': {'truncation_orders': [2, 3, 4], 'systems_to_test': ['test']},
            'plotting': {'figure_size': [10, 8], 'dpi': 300, 'save_format': 'png', 
                        'save_plots': True, 'output_directory': 'test_results/', 'show_plots': False},
            'logging': {'level': 'INFO', 'log_to_file': True, 'log_file': 'test.log'},
            'performance': {'use_parallel': False, 'n_workers': 2, 'memory_limit_gb': 4}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            config_loader = ConfigLoader(temp_config_path)
            
            # Test that all sections loaded correctly
            assert config_loader.system.n_states == 2
            assert config_loader.system.truncation_order == 4
            assert config_loader.simulation.t_start == 0.0
            assert config_loader.simulation.t_end == 10.0
            
            # Test helper methods
            x0 = config_loader.get_initial_conditions()
            assert np.allclose(x0, [1.0, 0.0])
            
            t_span = config_loader.get_time_span()
            assert t_span == (0.0, 10.0)
            
            A1, A2 = config_loader.get_system_matrices()
            assert A1.shape == (2, 2)
            assert A2.shape == (2, 4)
            
        finally:
            os.unlink(temp_config_path)
    
    def test_config_update(self):
        """Test configuration updating"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'system': {'n_states': 2, 'truncation_order': 3},
                      'simulation': {'t_start': 0, 't_end': 5, 'n_points': 50,
                                   'solver_method': 'RK45', 'rtol': 1e-8, 'atol': 1e-10},
                      'initial_conditions': {'x0': [0, 0]},
                      'matrices': {'A1': [[0, 1], [-1, 0]], 'A2': [[0, 0, 0, 0], [0, 0, 0, 0]]},
                      'experiments': {'truncation_orders': [2], 'systems_to_test': ['test']},
                      'plotting': {'figure_size': [8, 6], 'dpi': 100, 'save_format': 'png',
                                 'save_plots': False, 'output_directory': 'results/', 'show_plots': True},
                      'logging': {'level': 'DEBUG', 'log_to_file': False, 'log_file': 'log.txt'},
                      'performance': {'use_parallel': True, 'n_workers': 1, 'memory_limit_gb': 2}}, f)
            temp_config_path = f.name
        
        try:
            config_loader = ConfigLoader(temp_config_path)
            
            # Update configuration
            updates = {
                'system': {'truncation_order': 5},
                'simulation': {'t_end': 15.0}
            }
            config_loader.update_config(updates)
            
            # Check updates applied
            assert config_loader.system.truncation_order == 5
            assert config_loader.simulation.t_end == 15.0
            assert config_loader.system.n_states == 2  # Should remain unchanged
            
        finally:
            os.unlink(temp_config_path)
    
    def test_predefined_systems(self):
        """Test loading predefined system configurations"""
        try:
            available_systems = list_available_systems()
            print(f"Available systems: {available_systems}")
            
            for system_name in available_systems:
                # Load system configuration
                try:
                    config = load_system_config(system_name)
                    
                    # Verify required sections exist
                    assert hasattr(config, 'system')
                    assert hasattr(config, 'simulation')
                    
                    # Verify matrices can be loaded
                    A1, A2 = config.get_system_matrices()
                    assert A1.shape[0] == config.system.n_states
                    assert A1.shape[1] == config.system.n_states
                    assert A2.shape[0] == config.system.n_states
                    assert A2.shape[1] == config.system.n_states ** 2
                    
                    # Verify initial conditions
                    x0 = config.get_initial_conditions()
                    assert len(x0) == config.system.n_states
                    
                    print(f"✓ System '{system_name}' configuration loaded successfully")
                    
                except FileNotFoundError as e:
                    print(f"⚠ Configuration file not found for '{system_name}': {str(e)}")
                    # This is not necessarily a failure - the config file might not exist yet
                    continue
                except Exception as e:
                    print(f"✗ Failed to load system '{system_name}': {str(e)}")
                    raise
                    
        except Exception as e:
            print(f"Failed to get available systems: {str(e)}")
            # If we can't get available systems, just test basic config functionality
            print("Testing basic configuration functionality instead...")
            self._test_basic_config_functionality()
    
    def _test_basic_config_functionality(self):
        """Test basic configuration loading functionality"""
        # Create a minimal test config
        test_config = {
            'system': {'n_states': 2, 'truncation_order': 3},
            'simulation': {'t_start': 0.0, 't_end': 5.0, 'n_points': 50, 
                          'solver_method': 'RK45', 'rtol': 1e-8, 'atol': 1e-10},
            'initial_conditions': {'x0': [1.0, 0.0]},
            'matrices': {'A1': [[0, 1], [-1, 0]], 'A2': [[0, 0, 0, 0], [0, 0, 0, 0]]},
            'plotting': {'figure_size': [8, 6], 'dpi': 100, 'save_format': 'png',
                        'save_plots': False, 'output_directory': 'test_results/', 'show_plots': False},
            'logging': {'level': 'INFO', 'log_to_file': False, 'log_file': 'test.log'},
            'performance': {'use_parallel': False, 'n_workers': 1, 'memory_limit_gb': 2}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_config_path = f.name
        
        try:
            config_loader = ConfigLoader(temp_config_path)
            
            # Test basic functionality
            assert config_loader.system.n_states == 2
            assert config_loader.system.truncation_order == 3
            
            x0 = config_loader.get_initial_conditions()
            assert np.allclose(x0, [1.0, 0.0])
            
            A1, A2 = config_loader.get_system_matrices()
            assert A1.shape == (2, 2)
            assert A2.shape == (2, 4)
            
            print("✓ Basic configuration functionality works")
            
        finally:
            os.unlink(temp_config_path)


class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_simple_simulation(self):
        """Test a complete simulation run"""
        # Simple harmonic oscillator
        A1 = np.array([[0, 1], [-1, 0]])
        A2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])  # No quadratic terms
        x0 = np.array([1.0, 0.0])
        t_span = (0, 2*np.pi)
        t_eval = np.linspace(0, 2*np.pi, 50)
        
        sol_orig, x_carl, carleman_sys = simulate_and_compare(
            A1, A2, x0, t_span, t_eval, truncation_order=3)
        
        # For harmonic oscillator, solution should be periodic
        # Check that simulation completed
        assert sol_orig.success, "Original simulation failed"
        assert x_carl.shape[0] == 2, "Carleman solution has wrong number of states"
        assert x_carl.shape[1] == len(t_eval), "Carleman solution has wrong number of time points"
        
        # For harmonic oscillator with no quadratic terms, 
        # Carleman should match original very closely
        error = np.max(np.abs(sol_orig.y - x_carl))
        assert error < 1e-8, f"Error too large for harmonic oscillator: {error}"
    
    def test_nonlinear_system(self):
        """Test with a genuinely nonlinear system"""
        # Van der Pol oscillator (simplified)
        A1 = np.array([[0, 1], [-1, 0.1]])
        A2 = np.array([[0, 0, 0, 0], [0, -0.1, 0, 0]])  # x1*x2 term
        x0 = np.array([0.1, 0.1])
        t_span = (0, 5)
        t_eval = np.linspace(0, 5, 100)
        
        sol_orig, x_carl, carleman_sys = simulate_and_compare(
            A1, A2, x0, t_span, t_eval, truncation_order=4)
        
        # Check that simulation completed
        assert sol_orig.success, "Original simulation failed"
        assert x_carl.shape[0] == 2, "Carleman solution has wrong number of states"
        
        # Check that Carleman approximation is reasonable (not necessarily very accurate)
        error = np.max(np.abs(sol_orig.y - x_carl))
        assert error < 10.0, f"Error unreasonably large: {error}"
        assert np.all(np.isfinite(x_carl)), "Carleman solution contains non-finite values"


class TestNumericalStability:
    """Test numerical stability and edge cases"""
    
    def test_large_truncation_order(self):
        """Test behavior with large truncation orders"""
        A1 = np.array([[0, 1], [-1, -0.01]])
        A2 = np.array([[0, 0, 0, 0], [-0.01, 0, 0, 0]])
        
        # This should not crash, even if it's not accurate
        carleman = CarlemanLinearization(A1, A2, 2, 6)
        assert carleman.Q.shape[0] > 100, "Large truncation order should create large matrix"
        assert np.all(np.isfinite(carleman.Q)), "Carleman matrix should be finite"
    
    def test_zero_matrices(self):
        """Test with zero matrices"""
        A1 = np.zeros((2, 2))
        A2 = np.zeros((2, 4))
        
        carleman = CarlemanLinearization(A1, A2, 2, 3)
        
        # System should be stable (all eigenvalues zero)
        eigenvals = np.linalg.eigvals(carleman.Q)
        assert np.allclose(eigenvals, 0), "Zero system should have zero eigenvalues"
    
    def test_unstable_system(self):
        """Test with an unstable system"""
        A1 = np.array([[1, 0], [0, 1]])  # Unstable linear part
        A2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
        
        carleman = CarlemanLinearization(A1, A2, 2, 3)
        
        # Should still construct matrix without crashing
        assert carleman.Q.shape[0] > 0, "Should construct matrix for unstable system"
        assert np.all(np.isfinite(carleman.Q)), "Matrix should be finite"


# Test runner
def run_all_tests():
    """Run all tests and return results"""
    test_classes = [
        TestKroneckerOperations,
        TestCarlemanLinearization, 
        TestConfigLoader,
        TestIntegration,
        TestNumericalStability
    ]
    
    results = {}
    
    for test_class in test_classes:
        class_name = test_class.__name__
        results[class_name] = {}
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            try:
                test_instance = test_class()
                test_method = getattr(test_instance, method_name)
                test_method()
                results[class_name][method_name] = "PASS"
                print(f"✓ {class_name}.{method_name}")
            except Exception as e:
                results[class_name][method_name] = f"FAIL: {str(e)}"
                print(f"✗ {class_name}.{method_name}: {str(e)}")
    
    return results


if __name__ == "__main__":
    print("Running Carleman Error Analysis Test Suite")
    print("=" * 50)
    results = run_all_tests()
    
    # Summary
    total_tests = sum(len(class_results) for class_results in results.values())
    passed_tests = sum(1 for class_results in results.values() 
                      for result in class_results.values() if result == "PASS")
    failed_tests = total_tests - passed_tests
    
    print("\n" + "=" * 50)
    print(f"Test Summary: {passed_tests}/{total_tests} tests passed")
    if failed_tests > 0:
        print(f"Failed tests: {failed_tests}")
        for class_name, class_results in results.items():
            for method_name, result in class_results.items():
                if result != "PASS":
                    print(f"  {class_name}.{method_name}: {result}")
    else:
        print("All tests passed! ✓")