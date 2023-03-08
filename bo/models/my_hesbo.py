import sys 
sys.path.append("../")
from hesbo.count_sketch import RunMain
import numpy as np
LEGEND_SIZE = 15
from bo.functions import Levy

class HesBO:
    def __init__(self, f, lb, ub, low_dim, high_dim, n_init, max_evals, verbose):
       self.low_dim = low_dim 
       self.f = f 
       self.high_dim = high_dim  
       self.n_init = n_init 
       self.max_evals = max_evals
       self.box_size = max(np.max(np.abs(lb)) , np.max(np.abs(ub)))
       self.verbose = verbose
       self.maximising = True

    def optimize(self):
        best_results, elapsed, s, f_s, f_s_true, high_s = RunMain(low_dim=self.low_dim, high_dim=self.high_dim, initial_n=self.n_init, total_itr=self.max_evals, func_type='Branin', func=self.f,
            s=None, active_var=None, ARD=False, variance=1., length_scale=None, box_size=self.box_size, high_to_low=None, sign=None, hyper_opt_interval=20, noise_var=0, verbose=self.verbose)
        # print(best_results)
        # print(elapsed)
        self._X = s 
        self._fX = f_s
        # print( f_s_true)
        # print(high_s)

    @property
    def X(self):
        return self._X
    
    @property
    def fX(self):
        return self._fX

    
if __name__ == "__main__":
    hesbo = HesBO(Levy(10), 2,10,2, 10)
    hesbo.optimize()