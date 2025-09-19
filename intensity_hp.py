import numpy as np
from kernel import kernel


def intensity_hp(t, History, para):
    """
    Compute the intensity functions of Hawkes processes
    
    Args:
        t: Current time
        History: History of events [[times], [marks]]
        para: Parameters dictionary containing mu, A, and kernel parameters
        
    Returns:
        lambda_val: Intensity values for each dimension
    """
    lambda_val = para['mu'].copy()
    
    if History.size > 0:
        Time = History[0, :]
        Event = History[1, :]
        
        # Filter events that occurred before time t
        index = Time <= t
        Time = Time[index]
        Event = Event[index]
        
        if len(Time) > 0:
            # Compute kernel basis
            dt = t - Time
            basis = kernel(dt, para)
            
            # Convert events to 0-indexed for Python
            Event_idx = (Event - 1).astype(int)
            
            # Get A matrices for the events
            A = para['A'][Event_idx, :, :]  # Shape: (num_events, L, D)
            
            # Compute intensity contributions
            for c in range(para['A'].shape[2]):  # For each dimension
                contribution = np.sum(basis * A[:, :, c])
                lambda_val[c] += contribution
    
    # Ensure non-negative intensities
    lambda_val = lambda_val * (lambda_val > 0)
    
    return lambda_val


def sup_intensity_hp(t, History, para, options):
    """
    Compute the super bound of intensity function of Hawkes processes
    
    Args:
        t: Current time
        History: History of events [[times], [marks]]
        para: Parameters dictionary containing mu, A, and kernel parameters
        options: Options containing M (number of steps) and tstep
        
    Returns:
        mt: Maximum intensity bound
    """
    if History.size == 0:
        mt = np.sum(para['mu'])
    else:
        Time = History[0, :]
        Event = History[1, :]
        
        # Filter events that occurred before time t
        index = Time <= t
        Time = Time[index]
        Event = Event[index]
        
        if len(Time) == 0:
            mt = np.sum(para['mu'])
        else:
            MT = np.full(options['M'], np.sum(para['mu']))
            
            for m in range(options['M']):
                t_current = t + m * options['tstep'] / options['M']
                
                # Compute kernel basis
                dt = t_current - Time
                basis = kernel(dt, para)
                
                # Convert events to 0-indexed for Python
                Event_idx = (Event - 1).astype(int)
                
                # Get A matrices for the events
                A = para['A'][Event_idx, :, :]  # Shape: (num_events, L, D)
                
                # Compute intensity contributions
                for c in range(para['A'].shape[2]):  # For each dimension
                    contribution = np.sum(basis * A[:, :, c])
                    MT[m] += contribution
            
            mt = np.max(MT)
    
    # Ensure non-negative intensity
    mt = mt * (mt > 0)
    
    return mt