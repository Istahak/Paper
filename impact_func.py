import numpy as np
from kernel import kernel
from kernel_integration import kernel_integration


def impact_func(para, options):
    """
    Compute impact functions and infectivity matrix
    
    Args:
        para: Parameters dictionary containing A, kernel info
        options: Options dictionary containing M, dt, Tmax
        
    Returns:
        A: Infectivity matrix (D x D)
        Phi: Impact functions (D x M x D)
    """
    D1, L, D2 = para['A'].shape
    M = options['M']
    
    Phi = np.zeros((D1, M, D2))
    A = np.zeros((D2, D1))
    
    # Time stamps for computing impact functions
    time_stamp = np.arange(0, M * options['dt'], options['dt'])
    
    for u in range(D2):
        for v in range(D1):
            # Compute infectivity (integrated impact)
            basis_int = kernel_integration(options['Tmax'], para)
            A[u, v] = np.dot(para['A'][v, :, u], basis_int.flatten())
            
            # Compute impact function over time
            basis = kernel(time_stamp, para)
            Phi[v, :, u] = np.dot(para['A'][v, :, u], basis.T)
    
    return A, Phi