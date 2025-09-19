import numpy as np
from scipy.special import erf


def kernel_integration(dt, para):
    """
    Compute the integration of kernel function
    
    Args:
        dt: Time differences
        para: Parameters dictionary containing kernel, w, landmark
        
    Returns:
        G: Integrated kernel values
    """
    dt = np.array(dt).reshape(-1, 1)
    landmarks = np.array(para['landmark']).reshape(1, -1)
    
    # Calculate distance matrix
    distance = dt - landmarks
    landmark_matrix = np.tile(landmarks, (len(dt), 1))
    
    if para['kernel'] == 'exp':
        G = 1 - np.exp(-para['w'] * (distance - landmark_matrix))
        G[G < 0] = 0
        
    elif para['kernel'] == 'gauss':
        G = 0.5 * (erf(distance / (np.sqrt(2) * para['w'])) + 
                   erf(landmark_matrix / (np.sqrt(2) * para['w'])))
        
    else:
        raise ValueError('Error: please assign a kernel function!')
    
    return G