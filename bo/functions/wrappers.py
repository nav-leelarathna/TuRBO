
import numpy as np
from .robot_push_function import PushReward
from .rover_function import create_small_domain, create_large_domain
from emukit.core import ContinuousParameter, ParameterSpace
from abc import ABC, abstractmethod

class BaseFunction(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x) -> float:
        pass 

    @abstractmethod
    def getParameterSpace(self) -> ParameterSpace:
        pass

class Ackley(BaseFunction):
    def __init__(self, dim=10, sign=1):
        # Ackley is a minimisation function
        self.dim = dim  
        self.sign = sign
        self.lb = -5 * np.ones(self.dim)
        self.ub = 5 * np.ones(self.dim)

    def __call__(self, x):
        assert len(x) == self.dim
        x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
        n = len(x)
        s1 = sum( x**2 )
        s2 = sum( cos( c * x ))
        return self.sign * (-a*exp( -b*sqrt( s1 / n )) - exp( s2 / n ) + a + exp(1))
    
    def getParameterSpace(self) -> ParameterSpace:
        params = [ContinuousParameter(str(i),self.lb[i],self.ub[i]) for i in range(self.dim)]
        return ParameterSpace(params)

class Levy(BaseFunction):
    def __init__(self, dim=10, sign=1):
        # Levy is a minimisation problem, set sign depending on whether model is maximising or minimising
        self.sign = sign
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        
    def __call__(self, x):
        def f(x):
            w = 1 + (x - 1.0) / 4.0
            val = np.sin(np.pi * w[0]) ** 2 + \
                np.sum((w[1:self.dim - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:self.dim - 1] + 1) ** 2)) + \
                (w[self.dim - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[self.dim - 1])**2)
            return val
        if x.ndim == 1:
            return self.sign * f(x)
        ret =  self.sign * np.apply_along_axis(f, 1, x).reshape((-1,1))
        return ret
    
    def getParameterSpace(self):
        params = [ContinuousParameter(str(i),self.lb[i], self.ub[i]) for i in range(self.dim)]
        return ParameterSpace(params)

class RobotPush(BaseFunction):
    def __init__(self, sign=-1):
        # RobotPush is a maximisation problem, set sign depending on whether model is maximising or minimizing
        self.sign = sign
        self.f = PushReward()
        self.lb = np.array(self.f.xmin)
        self.ub = np.array(self.f.xmax)
        self.dim = len(self.f.xmin)

    def __call__(self, x):
        # flip sign to make it a minimisation problem
        if x.ndim == 1:
            return self.sign * self.f(x)
        ret =  self.sign * np.apply_along_axis(self.f, 1, x).reshape((-1,1))
        return ret

    def getParameterSpace(self):
        params = [ContinuousParameter(str(i),self.lb[i],self.ub[i]) for i in range(self.dim)]
        return ParameterSpace(params) 

class RoverControl(BaseFunction):
    def __init__(self, sign=-1):
        # RoverControl is a maximisation problem, set sign depending on whether model is maximising or minimizing
        self.sign = sign
        self.lb = None
        self.ub = None
        def l2cost(x, point):
            return 10 * np.linalg.norm(x - point, 1)
        self.f = create_large_domain(False,False, l2cost, l2cost)
        dim = self.f.traj.param_size
        self.lb = np.zeros(dim)
        self.ub = np.ones(dim)

    def __call__(self, x):
        # flip sign to make it a minimisation problem
        return self.sign * self.f(x)
    
    def getParameterSpace(self):
        params = [ContinuousParameter(str(i),self.lb[i],self.ub[i]) for i in range(self.dim)]
        return ParameterSpace(params)