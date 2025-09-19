import numpy as np
from scipy.stats import expon


def simulation_thinning_poisson(mu, t_start, t_end):
    """
    Implement thinning method to simulate homogeneous Poisson processes
    
    Args:
        mu: Intensity rates for each dimension
        t_start: Start time
        t_end: End time
        
    Returns:
        History: Array with [times, marks] where marks are 1-indexed
    """
    t = t_start
    History = []
    
    mu = np.array(mu)
    mt = np.sum(mu)
    
    while t < t_end:
        # Generate exponential random variable
        s = expon.rvs(scale=1/mt)
        t = t + s
        
        if t >= t_end:
            break
            
        # Generate random number for mark selection
        u = np.random.rand() * mt
        sum_intensities = 0
        
        for d in range(len(mu)):
            sum_intensities += mu[d]
            if sum_intensities >= u:
                break
        
        mark = d + 1  # Convert to 1-indexed for compatibility
        History.append([t, mark])
    
    if History:
        return np.array(History).T
    else:
        return np.array([[], []])