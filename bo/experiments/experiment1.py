import sys
sys.path.append('../')
from bo.functions import Levy, RobotPush, RoverControl
from bo.models import ModelFactory
from bo.utils import set_seed, getFunc
import numpy as np
import pandas as pd 
import os
import random
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
            "seed" : [0,1,2,3,4],
            "function" : ["RobotPush"],
            "model" : ["gp", "hesbo", "turbo1", "turboM"],
            "max_evals" : [2000],
            "batch_size" : [10],
            "n_init" : [5]  
        }
    }
    return sweep

if __name__ == "__main__":
    sweep = Sweep(robot_push_sweep())
    sweep.run()