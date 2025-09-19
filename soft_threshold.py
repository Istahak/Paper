import numpy as np


def soft_threshold_gs(A, thres):
    """
    Soft-thresholding for group lasso
    
    Reference:
    Yuan, Ming, and Yi Lin. 
    "Model selection and estimation in regression with grouped variables." 
    Journal of the Royal Statistical Society: Series B 
    (Statistical Methodology) 68.1 (2006): 49-67.
    
    Args:
        A: Input array of shape (D, L, D)
        thres: Threshold value
        
    Returns:
        Z: Group-sparse thresholded array
    """
    Z = np.zeros_like(A)
    
    for u in range(A.shape[2]):
        for v in range(A.shape[0]):
            norm_val = np.linalg.norm(A[v, :, u])
            if norm_val > 0:
                tmp = 1 - thres / norm_val
                if tmp > 0:
                    Z[v, :, u] = tmp * A[v, :, u]
    
    return Z


def soft_threshold_lr(A, thres):
    """
    Low-rank soft thresholding using SVD
    
    Args:
        A: Input array of shape (D, L, D)
        thres: Threshold value
        
    Returns:
        Z: Low-rank thresholded array
    """
    Z = np.zeros_like(A)
    
    for t in range(A.shape[1]):
        tmp = A[:, t, :].copy()
        U, S, Vt = np.linalg.svd(tmp, full_matrices=False)
        S = S - thres
        S[S < 0] = 0
        reconstructed = U @ np.diag(S) @ Vt
        Z[:, t, :] = reconstructed
    
    return Z