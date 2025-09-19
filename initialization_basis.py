import numpy as np
import random


def initialization_basis(Seqs, baseType=None, bandwidth=None, landmark=None):
    """
    Initialize model parameters for basis representation
    
    Args:
        Seqs: List of sequences
        baseType: Type of kernel basis (optional)
        bandwidth: Bandwidth parameter (optional)
        landmark: Landmark locations (optional)
        
    Returns:
        model: Dictionary containing initialized parameters
    """
    # Determine dimension
    D = 0
    for seq in Seqs:
        if len(seq.Mark) > 0:
            D = max(D, int(np.max(seq.Mark)))
    
    model = {}
    
    # Determine initialization case based on arguments
    if baseType is None:
        # Case 1: Default initialization
        sigma = np.zeros((D, D))
        Tmax = np.zeros((D, D))
        
        est = [[[] for _ in range(D)] for _ in range(D)]
        
        # Randomly sample sequences for initialization
        seq_indices = list(range(len(Seqs)))
        random.shuffle(seq_indices)
        
        for n in range(min(len(seq_indices), 10)):
            seq = Seqs[seq_indices[n]]
            if len(seq.Time) <= 1:
                continue
                
            for i in range(1, len(seq.Time)):
                ti = seq.Time[i]
                di = int(seq.Mark[i] - 1)  # Convert to 0-indexed
                
                for j in range(i):
                    tj = seq.Time[j]
                    dj = int(seq.Mark[j] - 1)  # Convert to 0-indexed
                    est[di][dj].append(ti - tj)
        
        # Calculate sigma and Tmax
        for di in range(D):
            for dj in range(D):
                if est[di][dj] and len(est[di][dj]) > 1:
                    data = np.array(est[di][dj])
                    std_val = np.std(data)
                    if std_val > 1e-6:  # Avoid zero std
                        sigma[di, dj] = ((4 * std_val**5) / (3 * len(data)))**0.2
                    else:
                        sigma[di, dj] = 1.0
                    Tmax[di, dj] = np.mean(data)
                else:
                    sigma[di, dj] = 1.0
                    Tmax[di, dj] = 5.0  # Default reasonable value
        
        Tmax_val = max(np.min(Tmax) / 2, 1.0)  # Ensure minimum value
        
        model['kernel'] = 'gauss'
        model['w'] = max(np.min(sigma) / 2, 0.5)  # Ensure minimum bandwidth
        max_landmarks = max(int(np.ceil(Tmax_val / model['w'])), 2)  # At least 2 landmarks
        model['landmark'] = model['w'] * np.arange(0, min(max_landmarks, 10) + 1)  # Cap at 10
        
    elif bandwidth is None:
        # Case 2: baseType provided only
        model['kernel'] = baseType
        
        sigma = np.zeros((D, D))
        Tmax = np.zeros((D, D))
        
        est = [[[] for _ in range(D)] for _ in range(D)]
        
        # Similar estimation as case 1
        seq_indices = list(range(len(Seqs)))
        random.shuffle(seq_indices)
        
        for n in range(min(len(seq_indices), 10)):
            seq = Seqs[seq_indices[n]]
            if len(seq.Time) <= 1:
                continue
                
            for i in range(1, len(seq.Time)):
                ti = seq.Time[i]
                di = int(seq.Mark[i] - 1)
                
                for j in range(i):
                    tj = seq.Time[j]
                    dj = int(seq.Mark[j] - 1)
                    est[di][dj].append(ti - tj)
        
        for di in range(D):
            for dj in range(D):
                if est[di][dj] and len(est[di][dj]) > 1:
                    data = np.array(est[di][dj])
                    std_val = np.std(data)
                    if std_val > 1e-6:
                        sigma[di, dj] = ((4 * std_val**5) / (3 * len(data)))**0.2
                    else:
                        sigma[di, dj] = 1.0
                    Tmax[di, dj] = np.mean(data)
                else:
                    sigma[di, dj] = 1.0
                    Tmax[di, dj] = 5.0
        
        Tmax_val = max(np.min(Tmax) / 2, 1.0)
        model['w'] = max(np.min(sigma) / 2, 0.5)
        max_landmarks = max(int(np.ceil(Tmax_val / model['w'])), 2)
        model['landmark'] = model['w'] * np.arange(0, min(max_landmarks, 10) + 1)
        
    elif landmark is None:
        # Case 3: baseType and bandwidth provided
        model['kernel'] = baseType
        model['w'] = bandwidth
        
        sigma = np.zeros((D, D))
        Tmax = np.zeros((D, D))
        
        est = [[[] for _ in range(D)] for _ in range(D)]
        
        seq_indices = list(range(len(Seqs)))
        random.shuffle(seq_indices)
        
        for n in range(min(len(seq_indices), 10)):
            seq = Seqs[seq_indices[n]]
            if len(seq.Time) <= 1:
                continue
                
            for i in range(1, len(seq.Time)):
                ti = seq.Time[i]
                di = int(seq.Mark[i] - 1)
                
                for j in range(i):
                    tj = seq.Time[j]
                    dj = int(seq.Mark[j] - 1)
                    est[di][dj].append(ti - tj)
        
        for di in range(D):
            for dj in range(D):
                if est[di][dj] and len(est[di][dj]) > 0:
                    data = np.array(est[di][dj])
                    Tmax[di, dj] = np.mean(data)
                else:
                    Tmax[di, dj] = 5.0
        
        Tmax_val = max(np.min(Tmax) / 2, 1.0)
        max_landmarks = max(int(np.ceil(Tmax_val / model['w'])), 2)
        model['landmark'] = model['w'] * np.arange(0, min(max_landmarks, 10) + 1)
        
    else:
        # Case 4: All parameters provided
        model['kernel'] = baseType
        model['w'] = bandwidth
        model['landmark'] = landmark
    
    # Initialize A and mu parameters
    L = len(model['landmark'])
    model['A'] = np.random.rand(D, L, D) / (D**2 * L)
    model['mu'] = np.random.rand(D) / D
    
    return model