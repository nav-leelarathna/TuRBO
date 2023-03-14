# from cma_es import CMA 
import cma 
from cmaes import CMA 
import numpy as np 

class CMA_ES:
    def __init__(self, f, lb,ub,mean, max_evals, batch_size, sigma=1.3):
        self.f =f 
        self.mean = np.array(mean) 
        self.sigma = sigma 
        self.max_evals = max_evals
        self.population_size = batch_size
        self.iterations = int(self.max_evals / self.population_size)
        self.maximising=False 
        self.bounds = np.array([lb,ub]).T 

    def optimize(self):
        assert len(self.mean) == self.f.dim
        optimizer = CMA(mean=self.mean, sigma=self.sigma, bounds=self.bounds, population_size=self.population_size)
        # optimizer = cma.CMAEvolutionStrategy(self.f.dim * [0], 1.3)
        self._X = [] 
        self._fX = []
        for iteration in range(self.iterations):
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                value = self.f(x)
                solutions.append((x, value))
                self._X.append(x)
                self._fX.append(value)
                # print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
            print(f"Iteration: {iteration*optimizer.population_size}, best value so far: {np.min(self._fX)}")
            optimizer.tell(solutions)
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