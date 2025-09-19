"""
Learning Granger causality for Hawkes processes by MLE based method

This is the main script for testing causality detection in Hawkes processes.
Converted from MATLAB to Python.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Import our custom modules
from simulation_branch_hp import simulation_branch_hp
from simulation_thinning_hp import simulation_thinning_hp
from initialization_basis import initialization_basis
from learning_mle_basis import learning_mle_basis
from impact_func import impact_func


def main():
    print("Learning Granger causality for Hawkes processes by MLE based method")
    
    # Clear workspace (Python equivalent)
    np.random.seed(42)  # For reproducibility
    
    # Options
    options = {
        'N': 2000,         # the number of sequences
        'Nmax': 500,       # the maximum number of events per sequence
        'Tmax': 200,       # the maximum size of time window
        'tstep': 0.1,
        'dt': 0.1,         # the length of each time step
        'M': 250,          # the number of steps in the time interval for computing sup-intensity
        'GenerationNum': 100  # the number of generations for branch processing
    }
    
    D = 7  # the dimension of Hawkes processes
    
    print('Approximate simulation of Hawkes processes via branching process')
    print('Complicated gaussian kernel')
    
    # Ground truth parameters
    para1 = {
        'kernel': 'gauss',        # the type of kernels per impact function
        'w': 2,                   # the bandwidth of gaussian kernel
        'landmark': np.arange(0, 13, 4)  # the central locations of kernels: 0:4:12
    }
    
    L = len(para1['landmark'])
    
    # Initialize ground truth parameters
    para1['mu'] = np.random.rand(D) / D
    para1['A'] = np.zeros((D, D, L))
    
    for l in range(L):
        para1['A'][:, :, l] = (0.5 ** (l + 1)) * (0.5 + np.ones((D, D)))
    
    mask = np.random.rand(D, D) * (np.random.rand(D, D) > 0.7).astype(float)
    para1['A'] = para1['A'] * mask[:, :, np.newaxis]
    
    # Ensure stationarity of Hawkes process
    eigenvals = np.linalg.eigvals(np.sum(para1['A'], axis=2))
    max_eigenval = np.max(np.abs(eigenvals))
    para1['A'] = 0.25 * para1['A'] / max_eigenval
    
    # Reshape A matrix to match expected format (D, L, D)
    tmp = para1['A'].copy()
    para1['A'] = np.zeros((D, L, D))
    for di in range(D):
        for dj in range(D):
            phi = tmp[di, dj, :]
            para1['A'][dj, :, di] = phi
    
    # Two simulation methods
    print("Simulating sequences using branching process...")
    Seqs1 = simulation_branch_hp(para1, options)
    # Alternative: Seqs1 = simulation_thinning_hp(para1, options)
    
    # Visualize all impact functions and infectivity matrix
    print("Computing impact functions...")
    A, Phi = impact_func(para1, options)
    
    print('Maximum likelihood estimation and basis representation')
    
    # Algorithm parameters
    alg1 = {
        'LowRank': 0,        # without low-rank regularizer
        'Sparse': 1,         # with sparse regularizer
        'alphaS': 1,
        'GroupSparse': 1,    # with group-sparse regularizer
        'alphaGS': 100,
        'outer': 8,
        'rho': 0.1,          # the initial parameter for ADMM
        'inner': 5,
        'thres': 1e-5,
        'Tmax': None,
        'storeErr': 0,
        'storeLL': 0
    }
    
    print("Initializing model...")
    model1 = initialization_basis(Seqs1)
    
    # Learning the model by MLE
    print("Learning model by MLE...")
    model1 = learning_mle_basis(Seqs1, model1, alg1)
    
    # Compute estimated impact functions
    A1, Phi1 = impact_func(model1, options)
    
    # Visualize the infectivity matrix (the adjacent matrix of Granger causality graph)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(A, cmap='viridis')
    plt.title('Ground truth of infectivity')
    plt.colorbar()
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.imshow(A1, cmap='viridis')
    plt.title('Estimated infectivity-MLE')
    plt.colorbar()
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\nResults:")
    print(f"Ground truth infectivity matrix shape: {A.shape}")
    print(f"Estimated infectivity matrix shape: {A1.shape}")
    print(f"Mean absolute error: {np.mean(np.abs(A - A1)):.6f}")
    print(f"Relative error: {np.linalg.norm(A - A1) / np.linalg.norm(A):.6f}")
    
    return {
        'ground_truth': {'A': A, 'Phi': Phi, 'para': para1},
        'estimated': {'A': A1, 'Phi': Phi1, 'model': model1},
        'sequences': Seqs1,
        'options': options
    }


if __name__ == "__main__":
    results = main()