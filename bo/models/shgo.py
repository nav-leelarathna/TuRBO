import scipy 
import numpy as np 
import random 

class SHGO:
    def __init__(self, f, lb, ub, max_evals ):
        self.f = f 
        self.max_evals = max_evals 
        self.bounds = [(lbi, ubi) for (lbi,ubi) in zip(lb, ub)]
        self.maximising = False
        self._X = []
        self._fX = []
        self.iters = 0

    def optimize(self):
        def callback(xk):
            self.iters += 1 
            print(self.iters)
            self._X.append(xk)
            self._fX.append(self.f(xk))
            if self.iters % 20 == 0:
                print(f"Iteration: {self.iters}, best value: {np.min(self._fX)}")
            # return False
        x0 = [random.uniform(lbi, ubi) for (lbi,ubi) in self.bounds]
        options = {
            # "maxfev" : self.max_evals,
            "maxfev" : self.max_evals,
            }
        scipy.optimize.shgo(func=self.f,bounds=self.bounds,n=100,callback=callback, options=options)
        self._fX = np.array(self._fX)
        self._X = np.array(self._X)

    @property
    def X(self):
        return self._X
    
    @property
    def fX(self):
        return self._fX
    