import sys 
sys.path.append("../")
from emukit.examples.gp_bayesian_optimization.single_objective_bayesian_optimization import GPBayesianOptimization
from emukit.test_functions import forrester_function
from emukit.core.loop.user_function import UserFunctionWrapper
from emukit.core import ContinuousParameter, ParameterSpace
import numpy as np
LEGEND_SIZE = 15
import matplotlib.pyplot as plt
from bo.turbo.utils import latin_hypercube, from_unit_cube
from emukit.examples.gp_bayesian_optimization.enums import AcquisitionType

class GP_BO:
    def __init__(self, f, lb, ub, n_init, max_evals, batch_size, acquisition_type=AcquisitionType.EI, noiseless=True):
        self.dim = len(lb)
        self.f = f 
        # self.max_evals = max_evals
        self.lb = lb 
        self.ub = ub 
        self.dim = len(lb)
        parameter_space = [ContinuousParameter(str(i),lb[i], ub[i]) for i in range(self.dim)]
        
        self.n_init = n_init
        X_init = latin_hypercube(self.n_init, self.dim)
        X_init = from_unit_cube(X_init, self.lb, self.ub)
        Y_init = self.f(X_init)
        self.bo = GPBayesianOptimization(variables_list=parameter_space,X=X_init, Y=Y_init, acquisition_type=acquisition_type, noiseless=noiseless, batch_size=batch_size)
        self.max_iterations = int(max_evals / batch_size)
        # TODO check if this is minimising or maximising
        self.maximising = True

    def optimize(self):
        self.bo.run_optimization(self.f, self.max_iterations)

    @property
    def X(self):
        return self.bo.loop_state.X
    
    @property
    def fX(self):
        return self.bo.loop_state.Y
    

    def predict(self, x):
        return self.bo.model.predict(x)

if __name__ == "__main__":
    target_function, space = forrester_function()
    x_plot = np.linspace(space.parameters[0].min, space.parameters[0].max, 200)[:, None]
    y_plot = target_function(x_plot)
    lb = np.array([p.min for p in space.parameters])
    ub = np.array([p.max for p in space.parameters])
    bo = GP_BO(target_function, lb, ub, 5, 50,5)
    bo.optimize()

    mu_plot, var_plot = bo.predict(x_plot)
    print(bo.X)
    print(bo.fX)

    plt.figure(figsize=(12, 8))
    plt.plot(bo.X, bo.fX, "ro", markersize=10, label="Observations")
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