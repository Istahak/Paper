# Hawkes Process Granger Causality Detection

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-green.svg)](https://scipy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation for learning Granger causality in Hawkes processes using Maximum Likelihood Estimation (MLE) with basis representation. Converted from MATLAB research code with full functionality and improved error handling.

## 🔬 Overview

Hawkes processes are self-exciting point processes where past events increase the probability of future events. This implementation learns causal relationships between different dimensions of multivariate Hawkes processes using advanced optimization techniques.

**Key Features:**

- 🚀 **Multiple Simulation Methods**: Branching process and Ogata's thinning algorithm
- 🔧 **Flexible Kernel Functions**: Gaussian and exponential kernels with customizable landmarks
- 📊 **Advanced Regularization**: Sparse, group-sparse, and low-rank regularization options
- ⚡ **ADMM Optimization**: Efficient parameter learning with convergence monitoring
- 🧪 **Comprehensive Testing**: Validation scripts with synthetic data generation
- 📖 **Research-Grade**: Based on peer-reviewed ICML 2016 paper

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/hawkes-process-causality.git
cd hawkes-process-causality
pip install -r requirements.txt
```

### Basic Usage

```python
# Run complete causality analysis
python3 test_causality.py

# Or use as a module
from test_causality import main
results = main()
```

### Simple Example

````python

- `test_causality.py` - Main script that runs the complete pipeline

### Core Modules

- `simulation_branch_hp.py` - Hawkes process simulation using branching process
- `simulation_thinning_hp.py` - Hawkes process simulation using thinning method
- `simulation_thinning_poisson.py` - Poisson process simulation
- `learning_mle_basis.py` - MLE learning with ADMM optimization
- `initialization_basis.py` - Model parameter initialization
- `loglike_basis.py` - Log-likelihood computation

### Utility Functions

- `kernel.py` - Kernel functions (Gaussian, Exponential)
- `kernel_integration.py` - Kernel integration functions
- `impact_function.py` - Impact function computation
- `impact_func.py` - Impact function and infectivity matrix computation
- `intensity_hp.py` - Intensity function computation
- `soft_threshold.py` - Soft thresholding for group-sparse and low-rank regularization
- `soft_threshold_s.py` - Soft thresholding for sparse regularization

## Installation

1. Install required packages:

```bash
pip install -r requirements.txt
````

## Usage

Run the main script:

```bash
python test_causality.py
```

This will:

1. Generate synthetic Hawkes process data with known ground truth
2. Initialize model parameters
3. Learn the causal structure using MLE with regularization
4. Visualize the results comparing ground truth vs estimated infectivity matrices

## Parameters

### Options

- `N`: Number of sequences (default: 2000)
- `Nmax`: Maximum events per sequence (default: 500)
- `Tmax`: Maximum time window (default: 200)
- `M`: Number of time steps for supremum intensity (default: 250)
- `GenerationNum`: Number of generations for branching process (default: 100)

### Algorithm Parameters

- `LowRank`: Enable low-rank regularization (0/1)
- `Sparse`: Enable sparse regularization (0/1)
- `GroupSparse`: Enable group-sparse regularization (0/1)
- `alphaS`: Sparse regularization parameter
- `alphaGS`: Group-sparse regularization parameter
- `outer`: Number of outer ADMM iterations
- `inner`: Number of inner iterations
- `rho`: ADMM penalty parameter

## Output

The script produces:

- Infectivity matrices showing causal relationships
- Performance metrics (mean absolute error, relative error)
- Visualization comparing ground truth vs estimated results

## Key Features

- **Multiple simulation methods**: Branching process and thinning methods
- **Flexible kernel functions**: Gaussian and exponential kernels
- **Regularization options**: Sparse, group-sparse, and low-rank regularization
- **ADMM optimization**: Efficient parameter learning
- **Basis representation**: Flexible modeling of impact functions
