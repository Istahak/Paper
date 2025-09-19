# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-09-19

### Added

- Complete MATLAB to Python conversion of Hawkes process causality detection
- Branching process simulation (`simulation_branch_hp.py`)
- Thinning algorithm simulation (`simulation_thinning_hp.py`)
- MLE learning with ADMM optimization (`learning_mle_basis.py`)
- Flexible kernel functions (Gaussian, exponential)
- Multiple regularization options (sparse, group-sparse, low-rank)
- Comprehensive test suite
- Documentation and examples

### Features

- Support for multivariate Hawkes processes
- Basis representation for impact functions
- Automatic parameter initialization
- Convergence monitoring and error handling
- Synthetic data generation capabilities

### Testing

- Simple functionality tests
- Minimal working demonstrations
- Comprehensive validation with 50 sequences
- Error handling and edge case coverage

### Documentation

- Complete README with usage examples
- API documentation in docstrings
- Conversion notes from MATLAB
- Contributing guidelines

### Performance

- Optimized NumPy/SciPy operations
- Efficient sparse matrix handling
- Fast simulation and learning algorithms
