import numpy as np
from kernel import kernel


def impact_function(u, dt, para):
    """
    Compute impact function values
    
    Args:
        u: Node index (1-indexed in MATLAB, 0-indexed in Python)
        dt: Time difference
        para: Parameters dictionary containing A array and kernel parameters
        
    Returns:
        phi: Impact function values
    """
    # Convert from 1-indexed (MATLAB) to 0-indexed (Python)
    u_idx = int(u - 1) if isinstance(u, (int, float)) else int(u[0] - 1)
    
    # Get the A matrix for this node: A[u, :, :]
    A = para['A'][u_idx, :, :]  # Shape: (L, D)
    
    # Compute basis functions
    basis = kernel(dt, para)  # Shape: (1, L) for single dt
    
    # If dt is scalar, basis will be (1, L), we need to handle this properly
    if basis.ndim == 2 and basis.shape[0] == 1:
        basis = basis.flatten()  # Shape: (L,)
    elif basis.ndim == 2:
        basis = basis[0, :]  # Take first row if multiple time points
    
    # Compute impact function: A.T @ basis gives us (D,)
    phi = A.T @ basis
    
    return phi