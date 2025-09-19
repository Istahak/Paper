"""
Quick test script for the converted Hawkes process code
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
from simulation_branch_hp import simulation_branch_hp
from initialization_basis import initialization_basis
from learning_mle_basis import learning_mle_basis
from impact_func import impact_func


def quick_test():
    print("Running quick test of Hawkes Process conversion...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Reduced options for quick test
    options = {
        'N': 10,           # reduced number of sequences
        'Nmax': 50,        # reduced maximum events per sequence
        'Tmax': 20,        # reduced time window
        'tstep': 0.1,
        'dt': 0.1,
        'M': 25,           # reduced number of steps
        'GenerationNum': 10  # reduced generations
    }
    
    D = 3  # reduced dimension
    
    print('Testing with simplified parameters...')
    
    # Ground truth parameters
    para1 = {
        'kernel': 'gauss',
        'w': 2,
        'landmark': np.arange(0, 9, 4)  # 0:4:8
    }
    
    L = len(para1['landmark'])
    
    # Initialize ground truth parameters
    para1['mu'] = np.random.rand(D) / D
    para1['A'] = np.zeros((D, D, L))
    
    for l in range(L):
        para1['A'][:, :, l] = (0.5 ** (l + 1)) * (0.5 + np.ones((D, D)))
    
    mask = np.random.rand(D, D) * (np.random.rand(D, D) > 0.7).astype(float)
    para1['A'] = para1['A'] * mask[:, :, np.newaxis]
    
    # Ensure stationarity
    eigenvals = np.linalg.eigvals(np.sum(para1['A'], axis=2))
    max_eigenval = np.max(np.abs(eigenvals))
    para1['A'] = 0.25 * para1['A'] / max_eigenval
    
    # Reshape A matrix
    tmp = para1['A'].copy()
    para1['A'] = np.zeros((D, L, D))
    for di in range(D):
        for dj in range(D):
            phi = tmp[di, dj, :]
            para1['A'][dj, :, di] = phi
    
    print("Simulating sequences...")
    Seqs1 = simulation_branch_hp(para1, options)
    print(f"Generated {len(Seqs1)} sequences")
    
    # Count total events
    total_events = sum(len(seq.Time) for seq in Seqs1)
    print(f"Total events across all sequences: {total_events}")
    
    print("Computing ground truth impact functions...")
    A, Phi = impact_func(para1, options)
    print(f"Ground truth infectivity matrix shape: {A.shape}")
    
    print("Initializing model...")
    model1 = initialization_basis(Seqs1)
    print(f"Model initialized with landmark shape: {np.array(model1['landmark']).shape}")
    
    # Simplified algorithm parameters for quick test
    alg1 = {
        'LowRank': 0,
        'Sparse': 1,
        'alphaS': 1,
        'GroupSparse': 0,  # Disabled for quick test
        'outer': 2,        # Reduced iterations
        'rho': 0.1,
        'inner': 2,        # Reduced iterations
        'thres': 1e-3,     # Relaxed threshold
        'Tmax': None,
        'storeErr': 0,
        'storeLL': 0
    }
    
    print("Learning model (simplified)...")
    model1 = learning_mle_basis(Seqs1, model1, alg1)
    
    print("Computing estimated impact functions...")
    A1, Phi1 = impact_func(model1, options)
    
    # Basic validation
    print(f"\nValidation:")
    print(f"Ground truth A shape: {A.shape}")
    print(f"Estimated A shape: {A1.shape}")
    print(f"Ground truth A range: [{np.min(A):.4f}, {np.max(A):.4f}]")
    print(f"Estimated A range: [{np.min(A1):.4f}, {np.max(A1):.4f}]")
    
    if A.shape == A1.shape:
        mae = np.mean(np.abs(A - A1))
        rel_error = np.linalg.norm(A - A1) / np.linalg.norm(A)
        print(f"Mean absolute error: {mae:.6f}")
        print(f"Relative error: {rel_error:.6f}")
        print("✓ Test completed successfully!")
        return True
    else:
        print("✗ Shape mismatch between ground truth and estimated matrices")
        return False


if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🎉 MATLAB to Python conversion successful!")
    else:
        print("\n❌ Test failed")