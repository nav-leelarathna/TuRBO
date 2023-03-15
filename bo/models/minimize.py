import scipy 
import numpy as np 
import random 

class Minimize:
    def __init__(self, f, lb, ub, max_evals, method="Nelder-Mead"):
        self.f = f 
        self.max_evals = max_evals 
        self.method = method
        self.lb = lb 
        self.ub = ub
        self.bounds = [(lbi, ubi) for (lbi,ubi) in zip(lb, ub)]
        self.maximising = False
        self._X = []
        self._fX = []
        self.iters = 0

    def optimize(self):
        def callback(xk):
            self.iters += 1 
            self._X.append(xk)
            self._fX.append(self.f(xk))
            if self.iters % 20 == 0:
                print(f"Iteration: {self.iters}, best value: {np.min(self._fX)}")
            # return False
        x0 = [random.uniform(lbi, ubi) for (lbi,ubi) in self.bounds]
        options = {
            # "maxfev" : self.max_evals,
            "maxiter" : self.max_evals
            }
        # constraints = []
        # for i in range(self.f.dim):
        #     def con1(x):
        #         return x[i] - self.lb[i] 
        #     def con2(x):
        #         return self.ub[i]- x[i]
        #     constraints.append({'type':'ineq', 'fun': con1})
        #     constraints.append({'type':'ineq', 'fun': con2})
        scipy.optimize.minimize(fun=self.f, x0=x0,method=self.method, bounds=self.bounds, callback=callback, options = options, tol=0.)
        self._fX = np.array(self._fX)
        self._X = np.array(self._X)

    @property
    def X(self):
        return self._X
    
    @property
    def fX(self):
        return self._fX
    