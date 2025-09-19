import numpy as np
from kernel import kernel
from kernel_integration import kernel_integration
from soft_threshold import soft_threshold_gs, soft_threshold_lr
from soft_threshold_s import soft_threshold_s
from loglike_basis import loglike_basis
import time


def learning_mle_basis(Seqs, model, alg):
    """
    Learning Hawkes processes via maximum likelihood estimation
    Different regularizers (low-rank, sparse, group sparse) of parameters and
    their combinations are considered, which are solved via ADMM.
    
    Reference:
    Xu, Hongteng, Mehrdad Farajtabar, and Hongyuan Zha. 
    "Learning Granger Causality for Hawkes Processes." 
    International Conference on Machine Learning (ICML). 2016.
    
    Args:
        Seqs: List of sequences
        model: Model parameters dictionary containing A, mu
        alg: Algorithm parameters
        
    Returns:
        model: Updated model parameters
    """
    # Initial parameters
    Aest = model['A'].copy()
    muest = model['mu'].copy()
    
    # Initialize ADMM variables
    if alg.get('LowRank', 0):
        UL = np.zeros_like(Aest)
        ZL = Aest.copy()
    
    if alg.get('Sparse', 0):
        US = np.zeros_like(Aest)
        ZS = Aest.copy()
    
    if alg.get('GroupSparse', 0):
        UG = np.zeros_like(Aest)
        ZG = Aest.copy()
    
    D = Aest.shape[0]
    
    if alg.get('storeLL', 0):
        model['LL'] = np.zeros(alg['outer'])
    
    if alg.get('storeErr', 0):
        model['err'] = np.zeros((alg['outer'], 3))
    
    start_time = time.time()
    
    for o in range(alg['outer']):
        rho = alg['rho'] * (1.1 ** (o + 1))
        
        for n in range(alg['inner']):
            NLL = 0  # negative log-likelihood
            
            Amu = np.zeros(D)
            Bmu = np.zeros(D)
            
            CmatA = np.zeros_like(Aest)
            AmatA = np.zeros_like(Aest)
            BmatA = np.zeros_like(Aest)
            
            # Add regularization terms
            if alg.get('LowRank', 0):
                BmatA += rho * (UL - ZL)
                AmatA += rho
            
            if alg.get('Sparse', 0):
                BmatA += rho * (US - ZS)
                AmatA += rho
            
            if alg.get('GroupSparse', 0):
                BmatA += rho * (UG - ZG)
                AmatA += rho
            
            # E-step: evaluate the responsibility using current parameters
            for c in range(len(Seqs)):
                if len(Seqs[c].Time) > 0:
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
                    
                    Amu += (Tstop - Tstart)
                    
                    dT = Tstop - Time
                    GK = kernel_integration(dT, model)
                    
                    Nc = len(Time)
                    
                    for i in range(Nc):
                        ui = int(Event[i] - 1)  # Convert to 0-indexed
                        
                        # Update BmatA
                        mask = (Aest[ui, :, :] > 0).astype(float)
                        for d in range(D):
                            BmatA[ui, :, d] += mask[:, d] * GK[i, :]
                        
                        ti = Time[i]
                        lambdai = muest[ui]
                        pii = muest[ui]
                        pij = None
                        
                        if i > 0:
                            tj = Time[:i]
                            uj = Event[:i].astype(int) - 1  # Convert to 0-indexed
                            
                            dt = ti - tj
                            gij = kernel(dt, model)
                            
                            auiuj = Aest[uj, :, ui]
                            pij = auiuj * gij
                            lambdai += np.sum(pij)
                        
                        NLL -= np.log(lambdai)
                        pii = pii / lambdai
                        
                        if i > 0 and pij is not None:
                            pij = pij / lambdai
                            if np.sum(pij) > 0:
                                for j in range(len(uj)):
                                    uuj = uj[j]
                                    CmatA[uuj, :, ui] -= pij[j, :]
                        
                        Bmu[ui] += pii
                    
                    NLL += (Tstop - Tstart) * np.sum(muest)
                    
                    # Convert Event to 0-indexed for Python
                    Event_idx = Event.astype(int) - 1
                    NLL += np.sum(GK * np.sum(Aest[Event_idx, :, :], axis=2))
                else:
                    print(f'Warning: Sequence {c} is empty!')
            
            # M-step: update parameters
            mu = Bmu / Amu
            
            if (not alg.get('Sparse', 0) and not alg.get('GroupSparse', 0) and 
                not alg.get('LowRank', 0)):
                A = -CmatA / BmatA
                A[np.isnan(A)] = 0
                A[np.isinf(A)] = 0
            else:
                # Solve quadratic equation
                discriminant = BmatA**2 - 4 * AmatA * CmatA
                discriminant[discriminant < 0] = 0
                A = (-BmatA + np.sqrt(discriminant)) / (2 * AmatA)
                A[np.isnan(A)] = 0
                A[np.isinf(A)] = 0
            
            # Check convergence
            Err = np.sum(np.abs(A - Aest)) / np.sum(np.abs(Aest))
            Aest = A.copy()
            muest = mu.copy()
            model['A'] = Aest
            model['mu'] = muest
            
            elapsed_time = time.time() - start_time
            print(f'Outer={o+1}, Inner={n+1}, Obj={NLL:.6f}, RelErr={Err:.6f}, Time={elapsed_time:.2f}sec')
            
            if Err < alg['thres'] or (o == alg['outer'] - 1 and n == alg['inner'] - 1):
                break
        
        # Store log-likelihood
        if alg.get('storeLL', 0):
            Loglike = loglike_basis(Seqs, model, alg)
            model['LL'][o] = Loglike
        
        # Calculate error
        if alg.get('storeErr', 0) and 'truth' in alg:
            Err = np.zeros(3)
            Err[0] = np.linalg.norm(model['mu'] - alg['truth']['mu']) / np.linalg.norm(alg['truth']['mu'])
            Err[1] = np.linalg.norm(model['A'] - alg['truth']['A']) / np.linalg.norm(alg['truth']['A'])
            combined_model = np.concatenate([model['mu'].flatten(), model['A'].flatten()])
            combined_truth = np.concatenate([alg['truth']['mu'].flatten(), alg['truth']['A'].flatten()])
            Err[2] = np.linalg.norm(combined_model - combined_truth) / np.linalg.norm(combined_truth)
            model['err'][o, :] = Err
        
        # Update ADMM variables
        if alg.get('LowRank', 0):
            threshold = alg['alphaLR'] / rho
            ZL = soft_threshold_lr(Aest + UL, threshold)
            UL = UL + (Aest - ZL)
        
        if alg.get('Sparse', 0):
            threshold = alg['alphaS'] / rho
            ZS = soft_threshold_s(Aest + US, threshold)
            US = US + (Aest - ZS)
        
        if alg.get('GroupSparse', 0):
            threshold = alg['alphaGS'] / rho
            ZG = soft_threshold_gs(Aest + UG, threshold)
            UG = UG + (Aest - ZG)
    
    return model