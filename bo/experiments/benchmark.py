import emukit
import numpy as np
from emukit.benchmarking.loop_benchmarking.benchmarker import Benchmarker
from emukit.benchmarking.loop_benchmarking.metrics import MinimumObservedValueMetric, TimeMetric
from emukit.examples.gp_bayesian_optimization.enums import ModelType, AcquisitionType
from emukit.examples.models.bohamiann import Bohamiann
from emukit.examples.gp_bayesian_optimization.optimization_loops import create_bayesian_optimization_loop
from emukit.examples.gp_bayesian_optimization.single_objective_bayesian_optimization import GPBayesianOptimization
import matplotlib.pyplot as plt
from emukit.test_functions.branin import branin_function
import random 
import sys
sys.path.append('../')
from bo.functions import RobotPush, Levy

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

# branin_fcn, parameter_space = branin_function()
# func = RobotPush(False)
func = Levy(10)
parameter_space = func.getParameterSpace()

loops = [
    # ('Random Forest', lambda loop_state: create_bayesian_optimization_loop(loop_state.X, loop_state.Y, parameter_space, AcquisitionType.EI, ModelType.RandomForest)),
    ('Bohammian', lambda loop_state: create_bayesian_optimization_loop(loop_state.X, loop_state.Y, parameter_space, AcquisitionType.EI, ModelType.BayesianNeuralNetwork)),
    ('Gaussian Process', lambda loop_state: GPBayesianOptimization(parameter_space.parameters, loop_state.X, loop_state.Y, acquisition_type=AcquisitionType.EI, noiseless=True))
]

n_repeats = 1 # 10
n_initial_data = 5
n_iterations = 5

metrics = [MinimumObservedValueMetric(), TimeMetric()]

benchmarkers = Benchmarker(loops, func, parameter_space, metrics=metrics)
benchmark_results = benchmarkers.run_benchmark(n_iterations=n_iterations, n_initial_data=n_initial_data,  n_repeats=n_repeats)


from emukit.benchmarking.loop_benchmarking.benchmark_plot import BenchmarkPlot
colours = ['m', 'c']
line_styles = ['-', '--']

metrics_to_plot = ['minimum_observed_value']
plots = BenchmarkPlot(benchmark_results, loop_colours=colours, loop_line_styles=line_styles, 
                      metrics_to_plot=metrics_to_plot)
plots.make_plot()

plt.show()