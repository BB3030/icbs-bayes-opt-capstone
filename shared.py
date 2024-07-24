from scipy.stats import norm
import numpy as np

def ProbImprovementAF(posterior_means:      np.ndarray,
                      posterior_stds:       np.ndarray,
                      best_observation:     float,
                      exploration_param:    float = 0):
    
    
    #normalise
    normed_inputs = (posterior_means - (best_observation + exploration_param))/posterior_stds

    #This will return a numpy array of the normal cdf for every mean/std_dev pair 
    return norm.cdf(normed_inputs, 0, 1)


def FormatOutputs(next_query: np.ndarray, af_type: str):    
    clipped = np.clip(next_query,0.000001, 0.999999)
    formatted = ['{:.6f}'.format(x) for x in clipped]
    print(f'{"-".join(formatted)}: {af_type}')

def Count_Submissions():
    return -8