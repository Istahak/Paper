"""
Comprehensive test with reasonable dataset size to demonstrate 
the Hawkes Process conversion working properly
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from simulation_branch_hp import simulation_branch_hp
from initialization_basis import initialization_basis
from learning_mle_basis import learning_mle_basis
from impact_func import impact_func


def comprehensive_test():
    """
    Test with reasonable dataset size to show the conversion working well
    """
    print("🚀 COMPREHENSIVE HAWKES PROCESS TEST")
    print("="*60)
    
    # Set seed for reproducible results
    np.random.seed(123)
    
    # Reasonable dataset size
    options = {
        'N': 50,           # 50 sequences
        'Nmax': 30,        # Max 30 events each
        'Tmax': 20,        # 20-unit time window
        'tstep': 0.2,
        'dt': 0.2, 
        'M': 50,
        'GenerationNum': 20
    }
    
    print(f"Dataset: {options['N']} sequences, max {options['Nmax']} events each")
    print(f"Time window: {options['Tmax']} units")
    
    # 3D system with clear causality pattern
    D = 3
    print(f"\n🔗 Setting up {D}D system with causal chain: 1→2→3")
    
    para = {
        'kernel': 'gauss',
        'w': 1.5,
        'landmark': np.array([0.0, 2.0, 4.0])  # 3 landmarks
    }
    
    # Ground truth: causal chain 1→2→3
    para['mu'] = np.array([0.15, 0.1, 0.05])  # Decreasing baseline activity
    para['A'] = np.zeros((D, len(para['landmark']), D))
    
    # Strong causality 1→2
    para['A'][0, 0, 1] = 0.3  # First landmark, strong effect
    para['A'][0, 1, 1] = 0.1  # Second landmark, weaker effect
    
    # Medium causality 2→3  
    para['A'][1, 0, 2] = 0.2
    
    # Weak self-excitation
    para['A'][0, 0, 0] = 0.1
    para['A'][1, 0, 1] = 0.1
    para['A'][2, 0, 2] = 0.1
    
    print(f"Ground truth base rates μ: {para['mu']}")
    print("Ground truth causal structure:")
    print("  1→2: Strong (0.3 + 0.1)")
    print("  2→3: Medium (0.2)")  
    print("  Self-excitation: Weak (0.1 each)")
    
    # Generate data
    print(f"\n📊 Generating {options['N']} sequences...")
    start_time = __import__('time').time()
    sequences = simulation_branch_hp(para, options)
    sim_time = __import__('time').time() - start_time
    
    # Analyze generated data
    total_events = sum(len(seq.Time) for seq in sequences if len(seq.Time) > 0)
    non_empty = sum(1 for seq in sequences if len(seq.Time) > 0)
    avg_events = total_events / max(non_empty, 1)
    
    print(f"✅ Generated in {sim_time:.2f}s")
    print(f"   Total events: {total_events}")
    print(f"   Non-empty sequences: {non_empty}/{len(sequences)}")
    print(f"   Average events per sequence: {avg_events:.1f}")
    
    # Show sample sequences
    print("\n📋 Sample sequences:")
    for i, seq in enumerate(sequences[:3]):
        if len(seq.Time) > 0:
            events = [f"{t:.1f}→{int(m)}" for t, m in zip(seq.Time[:5], seq.Mark[:5])]
            events_str = ", ".join(events)
            more = "..." if len(seq.Time) > 5 else ""
            print(f"   Seq {i+1}: [{events_str}{more}] ({len(seq.Time)} events)")
    
    # Compute ground truth infectivity
    print(f"\n🎯 Computing ground truth infectivity...")
    A_true, _ = impact_func(para, options)
    print(f"Ground truth infectivity matrix ({A_true.shape[0]}×{A_true.shape[1]}):")
    for i in range(A_true.shape[0]):
        row_str = "  " + "  ".join([f"{A_true[i,j]:6.3f}" for j in range(A_true.shape[1])])
        print(row_str)
    
    # Learn the model
    print(f"\n🧠 Learning model from data...")
    start_time = __import__('time').time()
    model = initialization_basis(sequences)
    init_time = __import__('time').time() - start_time
    
    print(f"✅ Initialized in {init_time:.2f}s")
    print(f"   Landmarks: {len(model['landmark'])} points")
    print(f"   Model shape: A{model['A'].shape}, μ{model['mu'].shape}")
    
    alg = {
        'LowRank': 0,
        'Sparse': 1,
        'alphaS': 5,       # Moderate sparsity
        'GroupSparse': 1, 
        'alphaGS': 10,     # Moderate group sparsity
        'outer': 5,        # More iterations
        'inner': 3,
        'rho': 0.1,
        'thres': 1e-4,
        'Tmax': None,
        'storeErr': 0,
        'storeLL': 0
    }
    
    start_time = __import__('time').time()
    model = learning_mle_basis(sequences, model, alg)
    learn_time = __import__('time').time() - start_time
    
    print(f"✅ Learning completed in {learn_time:.2f}s")
    
    # Get estimated results
    print(f"\n📈 Computing estimated infectivity...")
    A_est, _ = impact_func(model, options)
    print(f"Estimated infectivity matrix ({A_est.shape[0]}×{A_est.shape[1]}):")
    for i in range(A_est.shape[0]):
        row_str = "  " + "  ".join([f"{A_est[i,j]:6.3f}" for j in range(A_est.shape[1])])
        print(row_str)
    
    # Detailed analysis
    print(f"\n🔍 ANALYSIS:")
    print("-" * 40)
    
    # Overall error
    mae = np.mean(np.abs(A_true - A_est))
    rmse = np.sqrt(np.mean((A_true - A_est)**2))
    rel_error = np.linalg.norm(A_true - A_est) / np.linalg.norm(A_true)
    
    print(f"Error metrics:")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"  Relative Error: {rel_error:.4f}")
    
    # Causality detection analysis
    print(f"\nCausality detection:")
    
    # Check 1→2 causality
    true_12 = A_true[1, 0]  # Effect of 1 on 2
    est_12 = A_est[1, 0]
    print(f"  1→2: TRUE={true_12:.3f}, EST={est_12:.3f}, Error={abs(true_12-est_12):.3f}")
    
    # Check 2→3 causality  
    true_23 = A_true[2, 1]  # Effect of 2 on 3
    est_23 = A_est[2, 1]
    print(f"  2→3: TRUE={true_23:.3f}, EST={est_23:.3f}, Error={abs(true_23-est_23):.3f}")
    
    # Check non-causal links (should be small)
    true_13 = A_true[2, 0]  # Direct 1→3 (should be small)
    est_13 = A_est[2, 0]
    print(f"  1→3: TRUE={true_13:.3f}, EST={est_13:.3f} (should be small)")
    
    true_21 = A_true[0, 1]  # Reverse 2→1 (should be small)
    est_21 = A_est[0, 1]
    print(f"  2→1: TRUE={true_21:.3f}, EST={est_21:.3f} (should be small)")
    
    # Success criteria
    success_criteria = []
    
    # Criterion 1: Overall error is reasonable
    if rel_error < 0.5:
        success_criteria.append("✅ Overall error acceptable")
    else:
        success_criteria.append("⚠️  High overall error")
    
    # Criterion 2: Strong causal links detected
    if est_12 > 0.05:  # 1→2 should be detected
        success_criteria.append("✅ Strong causality 1→2 detected")
    else:
        success_criteria.append("⚠️  Weak detection of 1→2 causality")
    
    # Criterion 3: Causal order preserved  
    if est_12 > est_21:  # 1→2 should be stronger than 2→1
        success_criteria.append("✅ Causal direction 1→2 preserved")
    else:
        success_criteria.append("⚠️  Causal direction unclear")
    
    print(f"\nSuccess criteria:")
    for criterion in success_criteria:
        print(f"  {criterion}")
    
    # Overall assessment
    success_count = sum(1 for c in success_criteria if c.startswith("✅"))
    print(f"\n🏆 OVERALL RESULT: {success_count}/{len(success_criteria)} criteria met")
    
    if success_count >= 2:
        print("🎉 CONVERSION TEST SUCCESSFUL!")
        print("   The Python version successfully detects Hawkes process causality!")
    else:
        print("🔧 Partial success - may need parameter tuning for better results")
    
    print("="*60)
    return success_count >= 2


if __name__ == "__main__":
    comprehensive_test()