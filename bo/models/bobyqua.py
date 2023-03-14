from __future__ import print_function
import numpy as np
import pybobyqa
import random

class BOBYQA:
    def __init__(self, f, lb,ub, max_evals):
        self.f =f 
        self.max_evals = max_evals
        self.maximising=False 
        self.lb = lb 
        self.ub = ub 
        self._X = []
        self._fX = []

    def optimize(self):
        self.iters = 0
        def callback(x,f):
            self.iters += 1
            self._X.append(x)
            self._fX.append(f)
            if self.iters % 20 == 0:
                print(f"Iteration: {self.iters}, best value: {np.min(self._fX)}")
        x0 = [random.uniform(lbi, ubi) for (lbi,ubi) in zip(self.lb, self.ub)]
        soln = pybobyqa.solve(self.f, x0, bounds=(self.lb, self.ub),maxfun=self.max_evals,seek_global_minimum=True,print_progress=False,callback=callback)
        self._fX = np.array(self._fX)
        self._X = np.array(self._X)

    @property
    def X(self):
        return self._X
    
    @property
    def fX(self):
        return self._fX
    
    @property
    def fX_true(self):
        return self.f.trueFunctionValue(self._X)