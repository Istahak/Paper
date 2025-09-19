import numpy as np
from scipy.stats import expon
from simulation_thinning_poisson import simulation_thinning_poisson
from impact_function import impact_function
import time


def simulation_branch_hp(para, options):
    """
    Simulate Hawkes processes as Branch processes
    
    Reference:
    Møller, Jesper, and Jakob G. Rasmussen. 
    "Approximate simulation of Hawkes processes." 
    Methodology and Computing in Applied Probability 8.1 (2006): 53-64.
    
    Args:
        para: Parameters dictionary containing mu, A, kernel info
        options: Options dictionary containing N, Nmax, Tmax, GenerationNum
        
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
        # The 0-th generation: simulate exogenous events via Poisson processes
        History = simulation_thinning_poisson(para['mu'], 0, options['Tmax'])
        current_set = History
        
        for k in range(options['GenerationNum']):
            future_set = []
            
            if current_set.size == 0:
                break
                
            for i in range(current_set.shape[1]):
                ti = current_set[0, i]
                ui = current_set[1, i]
                t = 0
                
                phi_t = impact_function(ui, t, para)
                mt = np.sum(phi_t)
                
                while t < options['Tmax'] - ti and mt > 0:
                    # Generate exponential random variable
                    s = expon.rvs(scale=1/mt)
                    U = np.random.rand()
                    
                    phi_ts = impact_function(ui, t + s, para)
                    mts = np.sum(phi_ts)
                    
                    if t + s > options['Tmax'] - ti or U > mts / mt:
                        t = t + s
                    else:
                        u = np.random.rand() * mts
                        sum_intensities = 0
                        
                        for d in range(len(phi_ts)):
                            sum_intensities += phi_ts[d]
                            if sum_intensities >= u:
                                break
                        
                        index = d + 1  # Convert to 1-indexed
                        t = t + s
                        
                        if not future_set:
                            future_set = [[t + ti], [index]]
                        else:
                            future_set[0].append(t + ti)
                            future_set[1].append(index)
                    
                    phi_t = impact_function(ui, t, para)
                    mt = np.sum(phi_t)
                    if mt <= 0:
                        break
            
            if not future_set or (History.size > 0 and History.shape[1] > options['Nmax']):
                break
            else:
                current_set = np.array(future_set)
                if History.size == 0:
                    History = current_set
                else:
                    History = np.column_stack([History, current_set])
        
        # Sort events by time
        if History.size > 0:
            sort_indices = np.argsort(History[0, :])
            seq = Sequence()
            seq.Time = History[0, sort_indices]
            seq.Mark = History[1, sort_indices]
            seq.Start = 0
            seq.Stop = options['Tmax']
            
            # Filter events within time window
            valid_indices = seq.Time <= options['Tmax']
            seq.Time = seq.Time[valid_indices]
            seq.Mark = seq.Mark[valid_indices]
            
            Seqs.append(seq)
        else:
            seq = Sequence()
            seq.Time = np.array([])
            seq.Mark = np.array([])
            seq.Start = 0
            seq.Stop = options['Tmax']
            Seqs.append(seq)
        
        if (n + 1) % 10 == 0 or n + 1 == options['N']:
            elapsed_time = time.time() - start_time
            num_events = len(Seqs[n].Mark) if Seqs[n].Mark.size > 0 else 0
            print(f'#seq={n+1}/{options["N"]}, #event={num_events}, time={elapsed_time:.2f}sec')
    
    return Seqs