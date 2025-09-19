import numpy as np


def kernel(dt, para):
    """
    Compute the value of kernel function at different time
    
    Args:
        dt: Time differences
        para: Dictionary containing:
            - kernel: Type of kernel ('exp' or 'gauss')
            - w: Bandwidth parameter
            - landmark: Central locations of kernels
            
    Returns:
        g: Kernel values
    """
    dt = np.array(dt).reshape(-1, 1)
    landmarks = np.array(para['landmark']).reshape(1, -1)
    
    # Calculate distance matrix
    distance = dt - landmarks
    
    if para['kernel'] == 'exp':
        g = para['w'] * np.exp(-para['w'] * distance)
        g[g > 1] = 0
        
    elif para['kernel'] == 'gauss':
        g = np.exp(-(distance**2) / (2 * para['w']**2)) / (np.sqrt(2 * np.pi) * para['w'])
        
    else:
        raise ValueError('Error: please assign a kernel function!')
    
    return g