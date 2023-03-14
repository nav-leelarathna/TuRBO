import scipy 
import numpy as np 
import random 

class Random:
    def __init__(self, f, lb, ub, max_evals):
        self.f = f 
        self.max_evals = max_evals 
        self.bounds = [(lbi, ubi) for (lbi,ubi) in zip(lb, ub)]
        self.maximising = False
        self._X = []
        self._fX = []

    def optimize(self):
        for i in range(self.max_evals):
            x = np.array([random.uniform(lbi, ubi) for (lbi,ubi) in self.bounds])
            fx = self.f(x)
            self._X.append(x)
            self._fX.append(fx)
            if i % 20 == 0:
                print(f"Iteration: {i}, best value: {np.min(self._fX)}")
        self._fX = np.array(self._fX)
        self._X = np.array(self._X)

    @property
    def X(self):
        return self._X
    
    @property
    def fX(self):
        return self._fX
    