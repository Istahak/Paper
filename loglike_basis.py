import numpy as np
from kernel import kernel
from kernel_integration import kernel_integration
import time


def loglike_basis(Seqs, model, alg):
    """
    Compute log-likelihood for Hawkes processes with basis representation
    
    Args:
        Seqs: List of sequences
        model: Model parameters dictionary containing A, mu
        alg: Algorithm parameters
        
    Returns:
        Loglike: Negative log-likelihood value
    """
    # Get parameters
    Aest = model['A']
    muest = model['mu']
    
    Loglike = 0  # negative log-likelihood
    
    # E-step: evaluate the responsibility using the current parameters
    for c in range(len(Seqs)):
        Time = Seqs[c].Time
        Event = Seqs[c].Mark
        Tstart = Seqs[c].Start
        
        if 'Tmax' not in alg or alg['Tmax'] is None:
            Tstop = Seqs[c].Stop
        else:
            Tstop = alg['Tmax']
            indt = Time < alg['Tmax']
            Time = Time[indt]
            Event = Event[indt]
        
        if len(Time) == 0:
            continue
            
        dT = Tstop - Time
        GK = kernel_integration(dT, model)
        
        Nc = len(Time)
        
        for i in range(Nc):
            ui = int(Event[i] - 1)  # Convert to 0-indexed
            ti = Time[i]
            
            lambdai = muest[ui]
            
            if i > 0:
                tj = Time[:i]
                uj = Event[:i].astype(int) - 1  # Convert to 0-indexed
                
                dt = ti - tj
                gij = kernel(dt, model)
                
                # Get A matrix elements for the interactions
                auiuj = Aest[uj, :, ui]  # Shape: (i, L)
                pij = auiuj * gij  # Element-wise multiplication
                lambdai += np.sum(pij)
            
            Loglike -= np.log(lambdai)
        
        # Add integral terms
        Loglike += (Tstop - Tstart) * np.sum(muest)
        
        # Convert Event to 0-indexed for Python
        Event_idx = Event.astype(int) - 1
        Loglike += np.sum(GK * np.sum(Aest[Event_idx, :, :], axis=2))
    
    return -Loglike