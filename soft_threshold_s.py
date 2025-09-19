import numpy as np


def soft_threshold_s(A, thres):
    """
    Soft thresholding function for sparse regularization
    
    Args:
        A: Input array
        thres: Threshold value
        
    Returns:
        Z: Thresholded array
    """
    tmp = A.copy()
    S = np.sign(tmp)
    tmp = np.abs(tmp) - thres
    tmp[tmp <= 0] = 0
    Z = S * tmp
    return Z