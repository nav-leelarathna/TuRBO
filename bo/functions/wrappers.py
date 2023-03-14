
import numpy as np
from .robot_push_function import PushReward
from .rover_function import create_small_domain, create_large_domain
from emukit.core import ContinuousParameter, ParameterSpace
from abc import ABC, abstractmethod

class BaseFunction(ABC):
    @abstractmethod
    def __init__(self,sign, dim, noiseVar):
        self.noiseVar 
        self.sign = sign 
        self.dim = dim

    def __call__(self, x):
        value = self.trueFunctionValue(x)
        if self.noiseVar  > 0:
            return value + np.random.normal(0, self.noiseVar , value.shape)
        else:
            return value

    @abstractmethod 
    def trueFunctionValue(self, x) -> float:
        pass

    @abstractmethod
    def getParameterSpace(self) -> ParameterSpace:
        pass

    def setSign(self,sign):
        self.sign = sign

class Ackley(BaseFunction):
    def __init__(self, dim=10, sign=1, noiseVar=0.0):
        # Ackley is a minimisation function
        self.dim = dim  
        self.sign = sign
        self.lb = -5 * np.ones(self.dim)
        self.ub = 5 * np.ones(self.dim)
        self.maximising = False
        self.noiseVar =noiseVar
    
    def trueFunctionValue(self, x) -> float:
        def f(x):
            assert len(x) == self.dim
            x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
            n = len(x)
            a=20
            b=0.2
            c=2*np.pi 
            s1 = sum( x**2 )
            s2 = sum( np.cos( c * x ))
            return (-a*np.exp( -b*np.sqrt( s1 / n )) - np.exp( s2 / n ) + a + np.exp(1))
        if x.ndim == 1:
            return self.sign * f(x)
        ret =  self.sign * np.apply_along_axis(f, 1, x).reshape((-1,1))
        return ret
    
    def getParameterSpace(self) -> ParameterSpace:
        params = [ContinuousParameter(str(i),self.lb[i],self.ub[i]) for i in range(self.dim)]
        return ParameterSpace(params)

class Levy(BaseFunction):
    def __init__(self, dim=10, sign=1, noiseVar=0.0):
        # Levy is a minimisation problem, set sign depending on whether model is maximising or minimising
        self.sign = sign
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        self.maximising = False
        self.noiseVar = noiseVar
    
    def trueFunctionValue(self, x) -> float:
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
    def __init__(self, sign=-1, dim=None, noiseVar=0.0):
        # RobotPush is a maximisation problem, set sign depending on whether model is maximising or minimizing, 14D
        self.sign = sign
        self.f = PushReward()
        self.lb = np.array(self.f.xmin)
        self.ub = np.array(self.f.xmax)
        self.dim = len(self.f.xmin)
        self.maximising = True
        self.noiseVar = noiseVar

    def trueFunctionValue(self, x) -> float:
        # flip sign to make it a minimisation problem
        if x.ndim == 1:
            return self.sign * self.f(x)
        ret =  self.sign * np.apply_along_axis(self.f, 1, x).reshape((-1,1))
        return ret

    def getParameterSpace(self):
        params = [ContinuousParameter(str(i),self.lb[i],self.ub[i]) for i in range(self.dim)]
        return ParameterSpace(params) 

class RoverControl(BaseFunction):
    def __init__(self, sign=-1, dim=None, noiseVar=0.0):
        # RoverControl is a maximisation problem, set sign depending on whether model is maximising or minimizing, 60D
        self.sign = sign
        self.lb = None
        self.ub = None
        def l2cost(x, point):
            return 10 * np.linalg.norm(x - point, 1)
        self.f = create_large_domain(False,False, l2cost, l2cost)
        self.dim = self.f.traj.param_size
        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)
        self.maximising = True
        self.noiseVar = noiseVar

    def trueFunctionValue(self, x) -> float:
        # flip sign to make it a minimisation problem
        # return self.sign * self.f(x)
        if x.ndim == 1:
            return self.sign * self.f(x)
        ret =  self.sign * np.apply_along_axis(self.f, 1, x).reshape((-1,1))
        return ret
    
    def getParameterSpace(self):
        params = [ContinuousParameter(str(i),self.lb[i],self.ub[i]) for i in range(self.dim)]
        return ParameterSpace(params)
    
# class NoisyFunc(BaseFunction):
#     def __init__(self,sign,dim, noiseVar):
#         self.noiseVar = noiseVar
#         super.__init__(sign=sign, dim=dim)

#     def __call__(self, x):
#         return addNoise(super()(x), self.noiseVar)

