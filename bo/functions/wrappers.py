
import numpy as np
from .robot_push_function import PushReward
from .rover_function import create_small_domain, create_large_domain

class Levy:
    def __init__(self, dim=10, maximise=False):
        if maximise:
            self.sign = -1
        else:
            self.sign = 1
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 1 + (x - 1.0) / 4.0
        val = np.sin(np.pi * w[0]) ** 2 + \
            np.sum((w[1:self.dim - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:self.dim - 1] + 1) ** 2)) + \
            (w[self.dim - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[self.dim - 1])**2)
        return self.sign * val

class RobotPush:
    def __init__(self, maximise=True):
        if maximise:
            self.sign = -1
        else:
            self.sign = 1
        self.f = PushReward()
        self.lb = np.array(self.f.xmin)
        self.ub = np.array(self.f.xmax)

    def __call__(self, x):
        # flip sign to make it a minimisation problem
        return self.sign * self.f(x) 

class RoverControl:
    def __init__(self, maximise=True):
        if maximise:
            self.sign = -1
        else:
            self.sign = 1
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