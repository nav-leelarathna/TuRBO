import sys
sys.path.append("../")
from bo.turbo import TurboM, Turbo1
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from bo.functions import Levy, RoverControl, RobotPush, Ackley
from bo.models.cma_es import CMA_ES
from bo.models.my_hesbo import HesBO
from bo.models.minimize import Minimize
from bo.models.bobyqua import BOBYQA
from bo.models.shgo import SHGO
from bo.models.random import Random
import scipy
# f = Levy(100, noiseVar=.1)#,sign=-1)  
f = Ackley(200, noiseVar=.1)#,sign=-1)  
# f = RobotPush()
# f = RoverControl()

# model = TurboM(
#     f=f,  # Handle to objective function
#     lb=f.lb,  # Numpy array specifying lower bounds
#     ub=f.ub,  # Numpy array specifying upper bounds
#     n_init=10,  # Number of initial bounds from an Symmetric Latin hypercube design
#     max_evals=1000,  # Maximum number of evaluations
#     n_trust_regions=5,  # Number of trust regions
#     batch_size=10,  # How large batch size TuRBO uses
#     verbose=True,  # Print information from each batch
#     use_ard=True,  # Set to true if you want to use ARD for the GP kernel
#     max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
#     n_training_steps=50,  # Number of steps of ADAM to learn the hypers
#     min_cuda=1024,  # Run on the CPU for small datasets
#     device="cpu",  # "cpu" or "cuda"
#     dtype="float64",  # float64 or float32
# )

# model = Turbo1(
#     f=f,  # Handle to objective function
#     lb=f.lb,  # Numpy array specifying lower bounds
#     ub=f.ub,  # Numpy array specifying upper bounds
#     n_init=20,  # Number of initial bounds from an Latin hypercube design
#     max_evals = 200,  # Maximum number of evaluations
#     batch_size=1,  # How large batch size TuRBO uses
#     verbose=True,  # Print information from each batch
#     use_ard=True,  # Set to true if you want to use ARD for the GP kernel
#     max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
#     n_training_steps=50,  # Number of steps of ADAM to learn the hypers
#     min_cuda=1024,  # Run on the CPU for small datasets
#     device="cpu",  # "cpu" or "cuda"
#     dtype="float64",  # float64 or float32
# )

# model = CMA_ES(f=f,lb=f.lb, ub=f.ub, mean=[0.]*f.dim, max_evals=10000, batch_size=10,sigma= 1.3)
model = Random(f=f,lb=f.lb, ub=f.ub, max_evals=1000)
# model = HesBO(f=f, lb=f.lb, ub=f.ub, low_dim=4, high_dim=10, n_init=10, max_evals=100, verbose=True)
# model = Minimize(f=f, lb=f.lb, ub=f.ub,max_evals=1000, method="Nelder-Mead")
# model = SHGO(f=f, lb=f.lb, ub=f.ub,max_evals=1000)
# model = BOBYQA(f=f, lb=f.lb, ub=f.ub,max_evals=1000)
# bounds = [(lbi, ubi) for (lbi, ubi) in zip(f.lb, f.ub)]

model.optimize()

X = model.X  # Evaluated points
# print(X)
fX = model.fX  # Observed values
# fX_true = model.fX_true  # Observed values
ind_best = np.argmin(fX)
f_best, x_best = fX[ind_best], X[ind_best, :]

print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))

fig = plt.figure(figsize=(7, 5))
matplotlib.rcParams.update({'font.size': 16})
plt.plot(f.sign*fX, 'b.', ms=10)  # Plot all evaluated points as blue dots
# plt.plot(f.sign*fX_true , 'g.', ms=5)  # Plot all evaluated points as blue dots
plt.plot(f.sign*np.minimum.accumulate(fX), 'r', lw=3)  # Plot cumulative minimum as a red line
# plt.plot(f.sign*np.maximum.accumulate(fX), 'r', lw=3)  # Plot cumulative minimum as a red line
plt.xlim([0, len(fX)])
# plt.ylim([-10, 30])
plt.title("10D Levy function")

plt.tight_layout()
plt.show()