"""
Simple test script with minimal dataset for Hawkes Process conversion
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


def simple_test():
    print("🧪 Running simple test with minimal dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(123)
    
    # Very simple options for minimal test
    options = {
        'N': 5,            # Only 5 sequences
        'Nmax': 20,        # Max 20 events per sequence
        'Tmax': 10,        # Short time window
        'tstep': 0.2,
        'dt': 0.2,
        'M': 10,           # Few time steps
        'GenerationNum': 5  # Few generations
    }
    
    D = 2  # Only 2 dimensions for simplicity
    
    print(f'Testing with D={D} dimensions, {options["N"]} sequences...')
    
    # Simple ground truth parameters
    para1 = {
        'kernel': 'gauss',
        'w': 1.0,          # Simple bandwidth
        'landmark': np.array([0.0, 2.0])  # Just 2 landmarks
    }
    
    L = len(para1['landmark'])
    print(f'Using {L} landmarks: {para1["landmark"]}')
    
    # Initialize simple ground truth parameters
    para1['mu'] = np.array([0.1, 0.1])  # Simple base rates
    
    # Simple A matrix
    para1['A'] = np.zeros((D, L, D))
    para1['A'][0, 0, 1] = 0.2  # Node 0 affects node 1
    para1['A'][1, 1, 0] = 0.1  # Node 1 affects node 0 (weaker)
    
    print(f'Ground truth mu: {para1["mu"]}')
    print(f'Ground truth A shape: {para1["A"].shape}')
    print(f'Non-zero A elements: {np.sum(para1["A"] != 0)}')
    
    try:
        print("\n1️⃣ Testing impact function computation...")
        A_true, Phi_true = impact_func(para1, options)
        print(f"✅ Ground truth infectivity matrix computed: shape {A_true.shape}")
        print(f"Ground truth A:\n{A_true}")
        
        print("\n2️⃣ Testing simulation...")
        Seqs1 = simulation_branch_hp(para1, options)
        print(f"✅ Generated {len(Seqs1)} sequences")
        
        # Count total events
        total_events = sum(len(seq.Time) for seq in Seqs1 if len(seq.Time) > 0)
        print(f"Total events across all sequences: {total_events}")
        
        # Show some sequence details
        for i, seq in enumerate(Seqs1[:3]):  # Show first 3 sequences
            print(f"Seq {i+1}: {len(seq.Time)} events, times: {seq.Time[:5] if len(seq.Time) > 5 else seq.Time}")
        
        print("\n3️⃣ Testing model initialization...")
        model1 = initialization_basis(Seqs1)
        print(f"✅ Model initialized")
        print(f"Model landmarks: {model1['landmark']}")
        print(f"Model A shape: {model1['A'].shape}")
        print(f"Model mu: {model1['mu']}")
        
        print("\n4️⃣ Testing learning (very simple)...")
        # Very simple algorithm parameters
        alg1 = {
            'LowRank': 0,
            'Sparse': 0,       # No regularization for simplicity
            'GroupSparse': 0,
            'outer': 1,        # Just 1 outer iteration
            'rho': 0.1,
            'inner': 1,        # Just 1 inner iteration
            'thres': 1e-2,     # Loose threshold
            'Tmax': None,
            'storeErr': 0,
            'storeLL': 0
        }
        
        model1 = learning_mle_basis(Seqs1, model1, alg1)
        print("✅ Learning completed")
        
        print("\n5️⃣ Testing estimated impact functions...")
        A_est, Phi_est = impact_func(model1, options)
        print(f"✅ Estimated infectivity matrix computed: shape {A_est.shape}")
        print(f"Estimated A:\n{A_est}")
        
        # Basic comparison
        print(f"\n📊 Results:")
        print(f"Ground truth A:\n{A_true}")
        print(f"Estimated A:\n{A_est}")
        
        if A_true.shape == A_est.shape:
            mae = np.mean(np.abs(A_true - A_est))
            print(f"Mean absolute error: {mae:.6f}")
            
            # Check if we can detect the main causal relationship
            gt_max_idx = np.unravel_index(np.argmax(A_true), A_true.shape)
            est_max_idx = np.unravel_index(np.argmax(A_est), A_est.shape)
            print(f"Strongest ground truth connection: {gt_max_idx} = {A_true[gt_max_idx]:.4f}")
            print(f"Strongest estimated connection: {est_max_idx} = {A_est[est_max_idx]:.4f}")
            
            print("✅ Simple test completed successfully!")
            return True
        else:
            print("❌ Shape mismatch between ground truth and estimated matrices")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def create_manual_test_data():
    """Create a very simple manual test dataset"""
    print("\n🔧 Creating manual test data...")
    
    class SimpleSequence:
        def __init__(self, times, marks, start=0, stop=10):
            self.Time = np.array(times)
            self.Mark = np.array(marks)
            self.Start = start
            self.Stop = stop
            self.Feature = []
    
    # Create 3 simple sequences manually
    seqs = [
        SimpleSequence([1.0, 3.0, 5.0], [1, 2, 1]),      # Node 1 -> Node 2 -> Node 1
        SimpleSequence([0.5, 2.0, 4.5], [2, 1, 2]),      # Node 2 -> Node 1 -> Node 2  
        SimpleSequence([1.5, 6.0], [1, 2])               # Node 1 -> Node 2
    ]
    
    print("Manual sequences created:")
    for i, seq in enumerate(seqs):
        print(f"Seq {i+1}: times={seq.Time}, marks={seq.Mark}")
    
    # Test initialization with manual data
    try:
        print("\n🧪 Testing initialization with manual data...")
        model = initialization_basis(seqs)
        print(f"✅ Initialization successful")
        print(f"Landmarks: {model['landmark']}")
        print(f"A shape: {model['A'].shape}")
        return seqs, model
    except Exception as e:
        print(f"❌ Error with manual data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    # First try the simple simulation test
    print("=" * 60)
    print("🚀 SIMPLE HAWKES PROCESS TEST")
    print("=" * 60)
    
    success = simple_test()
    
    # If that fails, try manual data
    if not success:
        print("\n" + "=" * 60)
        print("🔧 MANUAL DATA TEST")
        print("=" * 60)
        seqs, model = create_manual_test_data()
        
        if seqs is not None:
            print("✅ Manual test data creation successful!")
        else:
            print("❌ Manual test failed")
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 CONVERSION TEST SUCCESSFUL!")
    else:
        print("🔍 DEBUGGING NEEDED")
    print("=" * 60)