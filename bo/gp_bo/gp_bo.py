from emukit.examples.gp_bayesian_optimization.single_objective_bayesian_optimization import GPBayesianOptimization
from emukit.test_functions import forrester_function
from emukit.core.loop.user_function import UserFunctionWrapper
from emukit.core import ContinuousParameter, ParameterSpace
import numpy as np
LEGEND_SIZE = 15
import matplotlib.pyplot as plt
from bo.turbo.utils import latin_hypercube

class GP_BO:
    def __init__(self, f, lb, ub, n_init, max_evals):
        self.dim = len(lb)
        self.f = f 
        self.max_evals = max_evals
        self.lb = lb 
        self.ub = ub 
        self.n_init = n_init


    def optimize(self):
        X_init = latin_hypercube(self.n_init, self.dim)
        X_init = from_unit_cube(X_init, self.lb, self.ub)
        Y_init = self.f(X_init)
        bo = GPBayesianOptimization(variables_list=[ContinuousParameter('x1', 0, 1)],X X_init, Y=Y_init)


target_function, space = forrester_function()
x_plot = np.linspace(space.parameters[0].min, space.parameters[0].max, 200)[:, None]
y_plot = target_function(x_plot)
X_init = np.array([[0.2],[0.6], [0.9]])
Y_init = target_function(X_init)
bo = GPBayesianOptimization(variables_list=[ContinuousParameter('x1', 0, 1)],
                            X=X_init, Y=Y_init)
bo.run_optimization(target_function, 20)

mu_plot, var_plot = bo.model.predict(x_plot)
print(bo.loop_state.X)
print(bo.loop_state.Y)

plt.figure(figsize=(12, 8))
plt.plot(bo.loop_state.X, bo.loop_state.Y, "ro", markersize=10, label="Observations")
plt.plot(x_plot, y_plot, "k", label="Objective Function")
plt.plot(x_plot, mu_plot, "C0", label="Model")
plt.fill_between(x_plot[:, 0],
                 mu_plot[:, 0] + np.sqrt(var_plot)[:, 0],
                 mu_plot[:, 0] - np.sqrt(var_plot)[:, 0], color="C0", alpha=0.6)

plt.fill_between(x_plot[:, 0],
                 mu_plot[:, 0] + 2 * np.sqrt(var_plot)[:, 0],
                 mu_plot[:, 0] - 2 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.4)

plt.fill_between(x_plot[:, 0],
                 mu_plot[:, 0] + 3 * np.sqrt(var_plot)[:, 0],
                 mu_plot[:, 0] - 3 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.2)
plt.legend(loc=2, prop={'size': LEGEND_SIZE})
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x)$")
plt.grid(True)
plt.xlim(0, 1)
plt.show()