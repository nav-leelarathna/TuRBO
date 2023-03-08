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

def generate_instances(sweep):
    configs = sweep["configurations"]
    return [(s,f,m, me, bs, n) for f in configs["function"] for m in configs["model"] for s in configs["seed"] for me in configs["max_evals"] for bs in configs["batch_size"]for n in configs["n_init"]]

def run_instance(func, model, seed, max_evals, batch_size, n_init):
    set_seed(seed)
    func = getFunc(func)
    space = func.getParameterSpace()
    lb = np.array([p.min for p in space.parameters])
    ub = np.array([p.max for p in space.parameters])
    factory = ModelFactory(func, lb, ub, n_init)
    match model:
        case "gp_bo":
            model = factory.getGP_BO(batch_size, max_evals)
        case "turbo1":
            model = factory.getTurbo1(batch_size, max_evals)
        case "turboM":
            model = factory.getTurboM(batch_size, max_evals)
        case _:
            raise Exception(f"Unknown model: {model}")
    model.optimize()
    X = model.X 
    fX = model.fX 
    numEvaluations = np.array(list(range(X.size)))
    return pd.DataFrame(list(zip(numEvaluations,X,fX)), columns=["evals","X", "fX"])

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

def sweep():
    sweep = {
        "name" : "robotPush",
        "configurations": {
            "seed" : [0,1,2,3,4],
            "function" : ["RobotPush"],
            "model" : ["turbo1", "turboM", "gp_bo", "hesbo"],
            "max_evals" : [1000],
            "batch_size" : [10],
            "n_init" : [5]  
        }
    }
    return sweep

def evaluateSweep(sweep_config):
    name = sweep["name"]
    datapath = f"data/{name}/sweep_results"
    os.makedirs(datapath, exist_ok=True)

    colNames = list(sweep["configurations"].keys()) + ["datapath"]
    results = []
    instanceConfigs = generate_instances(sweep_config)
    random.shuffle(instanceConfigs)

    resultsTableFilepath = f"data/{name}/results.csv"
    for j, instanceConfig in enumerate(instanceConfigs):
        print(f"Starting run {i}/{len(instanceConfigs)}")
        print(instanceConfig)
        filename = f"data/{name}/sweep_results/" + "_".join([str(i) for i in instanceConfig]) + ".csv" 
        instance_results = run_instance(*instanceConfig)
        instance_results.to_csv(filename)
        row = [instanceConfig[i] for i in range(len(instanceConfig))] + [filename]
        results.append(row)   
        resultsDF = pd.DataFrame(data=results, columns=colNames)
        resultsDF.to_csv(resultsTableFilepath)
        return


if __name__ == "__main__":
    # sweep = sweep()
    sweep = Sweep(sweep())
    sweep.run()
    # evaluateSweep(sweep)