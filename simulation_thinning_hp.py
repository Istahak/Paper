import numpy as np
from scipy.stats import expon
from intensity_hp import intensity_hp, sup_intensity_hp
import time


def simulation_thinning_hp(para, options):
    """
    Implementation of Ogata's thinning method to simulate Hawkes processes
    
    Reference:
    Ogata, Yosihiko. "On Lewis' simulation method for point processes." 
    IEEE Transactions on Information Theory 27.1 (1981): 23-31.
    
    Args:
        para: Parameters dictionary containing mu, A, kernel info
        options: Options dictionary containing N, Nmax, Tmax, M, tstep
        
    Returns:
        Seqs: List of sequences, each with Time, Mark, Start, Stop attributes
    """
    
    class Sequence:
        def __init__(self):
            self.Time = []
            self.Mark = []
            self.Start = 0
            self.Stop = 0
            self.Feature = []
    
    Seqs = []
    start_time = time.time()
    
    for n in range(options['N']):
        t = 0
        History = np.array([[], []])
        
        mt = sup_intensity_hp(t, History, para, options)
        
        while t < options['Tmax'] and (History.size == 0 or History.shape[1] < options['Nmax']) and mt > 0:
            # Generate exponential random variable
            s = expon.rvs(scale=1/mt)
            U = np.random.rand()
            
            # Compute intensity at proposed time
            lambda_ts = intensity_hp(t + s, History, para)
            mts = np.sum(lambda_ts)
            
            if t + s > options['Tmax'] or U > mts / mt:
                t = t + s
            else:
                # Accept the event and determine which dimension
                u = np.random.rand() * mts
                sum_intensities = 0
                
                for d in range(len(lambda_ts)):
                    sum_intensities += lambda_ts[d]
                    if sum_intensities >= u:
                        break
                
                index = d + 1  # Convert to 1-indexed
                t = t + s
                
                # Add event to history
                if History.size == 0:
                    History = np.array([[t], [index]])
                else:
                    History = np.column_stack([History, [t, index]])
            
            # Update supremum intensity
            mt = sup_intensity_hp(t, History, para, options)
            if mt <= 0:
                break
        
        # Create sequence object
        seq = Sequence()
        if History.size > 0:
            seq.Time = History[0, :]
            seq.Mark = History[1, :]
        else:
            seq.Time = np.array([])
            seq.Mark = np.array([])
        
        seq.Start = 0
        seq.Stop = options['Tmax']
        
        # Filter events within time window
        if len(seq.Time) > 0:
            valid_indices = seq.Time <= options['Tmax']
            seq.Time = seq.Time[valid_indices]
            seq.Mark = seq.Mark[valid_indices]
        
        Seqs.append(seq)
        
        if (n + 1) % 10 == 0 or n + 1 == options['N']:
            elapsed_time = time.time() - start_time
            num_events = len(seq.Time) if len(seq.Time) > 0 else 0
            print(f'#seq={n+1}/{options["N"]}, #event={num_events}, time={elapsed_time:.2f}sec')
    
    return Seqs