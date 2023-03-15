import sys
sys.path.append('../')
from bo.functions import Levy, RobotPush, RoverControl
from bo.models import ModelFactory
from bo.utils import set_seed, getFunc
import numpy as np
import pandas as pd 
from bo.experiments.sweep import Sweep
'''
Things to change
trust_regions
batch_size 
dimensionality of problem
seed

gp_bo uses a matern 5 2 kernel

We want to evaluate whether using trust regions does truly give us a benefit compared to normal BO, and also in setting where total number of runs is equal, i.e. increasing batch size.

Secondly, I hypothesize that the benefit gained by the bandits diminished quickly as the dimensionality of the problem increases, test by ablating over the dimensionality of the problem.

Replicate Figure 7 - shows why single agent BO only explores and optimises within a highly local volume of the search space

Replicate Figure 7 - showing that increasing batch size gives us a linear improvement, plotting reward vs steps

Prelim - Looks like regular GP BO works better with Levy(10) than Turbo1. Kind of obvious since a single trust regions spans a smaller space than the global space that GP BO looks at 

GP_BO, Bohammian, Turbo1, Turbo2
Wendel's theorem - limitation
'''

def _sweep():
    sweep = {
        "name" : "test",
        "configurations": {
            "seed" : [0,1,2,3,4],
            "function" : ["Levy_10", "RoverControl", "RobotPush", "Ackley_10"],
            "model" : ["turbo1", "turboM", "gp_bo", "hesbo"],
            "max_evals" : [1000],
            "batch_size" : [10],
            "n_init" : [5]  
        }
    }
    return sweep

def robot_push_sweep():
    sweep = {
        "name" : "robotPush",
        "configurations": {
            "seed" : [i for i in range(5)],
            "function" : ["RobotPush"],
            "model" : ["hesbo", "turbo1", "turboM"],#, "gp"],
            "max_evals" : [1000],
            "batch_size" : [10],
            "n_init" : [5]  
        }
    }
    return sweep

def sweep_100D():
    sweepConfig = {
        "name" : "100D",
        "configurations": {
            "seed" : [i for i in range(5)],
            "function" : ["Levy_100", "Ackley_100"],
            "model" : ["random","nelder-mead", "cobyla", "cmaes", "hesbo", "turbo1", "turboM"],#, "gp"],
            "max_evals" : [1000],
            "batch_size" : [10],
            "n_init" : [20],
            "noise" : [0.0]  
        }
    }
    sweep = Sweep(sweepConfig)
    sweep.run()

def sweep_turbo_dimensionality():
    sweepConfig = {
        "name" : "turbo_dimensionality2",
        "configurations": {
            "seed" : [i for i in range(5)],
            "function" : ["Ackley_4","Ackley_8", "Ackley_16", "Ackley_32"],
            "model" : ["turbo10", "turbo5", "turbo1"],
            "max_evals" : [2000],
            "batch_size" : [50],
            "n_init" : [20],
            "noise" : [0.0]  
        }
    }
    sweep = Sweep(sweepConfig)
    sweep.run()

def sweep_turbo_batch_size():
    sweepConfig = {
        "name" : "turbo_batch_size",
        "configurations": {
            "seed" : [i for i in range(5)],
            "function" : ["Levy_16"],
            "model" : ["turbo1","turbo4","turbo8","turbo16"],
            "max_evals" : [3000],
            "batch_size" : [4,8,16],
            "n_init" : [20],
            "noise" : [0.0]  
        }
    }
    sweep = Sweep(sweepConfig)
    sweep.run()

def sweep_problem_ablation():
    sweepConfig = {
        "name" : "problem_ablation",
        "configurations": {
            "seed" : [i for i in range(5,10)],
            "function" : ["RoverControl", "RobotPush", "Ackley_30", "Levy_10"],
            "model" : ["random","nelder-mead", "cmaes", "hesbo", "turbo1", "turbo20"],
            "max_evals" : [1000],
            "batch_size" : [20],
            "n_init" : [20],
            "noise" : [0.0]     
        }
    }
    sweep = Sweep(sweepConfig)
    sweep.run()
    # sweep.mock_results_file()

def sweep_turbo_m():
    sweepConfig = {
        "name" : "turbo1vs20",
        "configurations": {
            "seed" : [i for i in range(5)],
            "function" : ["RoverControl", "RobotPush", "Ackley_30", "Levy_10"],
            "model" : ["turbo1", "turbo20"],
            "max_evals" : [2000],
            "batch_size" : [50],
            "n_init" : [50],
            "noise" : [0.0]     
        }
    }
    sweep = Sweep(sweepConfig)
    sweep.run() 

def sweep_problem_ablation_rerun():
    sweepConfig = {
        "name" : "sweep_problem_ablation_rerun",
        "configurations": {
            "seed" : [i for i in range(5)],
            "function" : ["RoverControl","Ackley_30", "RobotPush", "Levy_10"],
            "model" : ["nelder-mead", "hesbo"],
            "max_evals" : [1000],
            "batch_size" : [10],
            "n_init" : [20],
            "noise" : [0.0]  
        }
    }
    sweep = Sweep(sweepConfig)
    sweep.run()

if __name__ == "__main__":
    globals()[sys.argv[1]]()
    # problems = ["RobotPush", "RoverControl", "Levy_10","Ackley_10",]
    # for p in problems:
    #     sweepConfig = robot_push_sweep()
    #     sweepConfig["configurations"]["function"] = [p]
    #     sweepConfig["name"] = p
    #     sweep = Sweep(sweepConfig)
    #     sweep.run()