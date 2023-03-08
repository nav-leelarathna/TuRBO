
from bo.utils import set_seed, getFunc
import sys
sys.path.append('../')
from bo.models import ModelFactory
import numpy as np
import pandas as pd 
import os
import time

class Sweep:
    def __init__(self,sweepConfig):
        self.sweepConfig = sweepConfig 

    def generate_instances(self):
        configs = self.sweepConfig["configurations"]
        return [(s,f,m, me, bs, n) for f in configs["function"] for m in configs["model"] for s in configs["seed"] for me in configs["max_evals"] for bs in configs["batch_size"]for n in configs["n_init"]]

    def run_instance(self, seed, func, model, max_evals, batch_size, n_init):
        set_seed(seed)
        func = getFunc(func)
        space = func.getParameterSpace()
        lb = np.array([p.min for p in space.parameters])
        ub = np.array([p.max for p in space.parameters])
        factory = ModelFactory(func, lb, ub, n_init)
        match model:
            case "gp":
                model = factory.getGP_BO(batch_size, max_evals)
            case "turbo1":
                model = factory.getTurbo1(batch_size, max_evals)
            case "turboM":
                model = factory.getTurboM(batch_size, max_evals)
            case "hesbo":
                model = factory.getHesbo(low_dim=4, max_evals=max_evals)
            case _:
                raise Exception(f"Unknown model: {model}")
        if func.maximising == model.maximising:
            func.setSign(1)
        else:
            func.setSign(-1)
        model.optimize()
        X = model.X 
        fX = model.fX 
        numEvaluations = np.array(list(range(X.size)))
        return pd.DataFrame(list(zip(numEvaluations,X,fX)), columns=["evals","X", "fX"])

    def run(self):
        name = self.sweepConfig["name"]
        datapath = f"data/{name}/sweep_results"
        os.makedirs(datapath, exist_ok=True)

        colNames = list(self.sweepConfig["configurations"].keys()) + ["datapath", "compute_time"]
        results = []
        instanceConfigs = self.generate_instances()
        # random.shuffle(instanceConfigs)

        resultsTableFilepath = f"data/{name}/results.csv"
        for j, instanceConfig in enumerate(instanceConfigs):
            # maybe add functionality to skip if instance already contained in results.csv so that we can restart 
            print(f"Starting run {j+1}/{len(instanceConfigs)}")
            print(instanceConfig)
            filename = f"data/{name}/sweep_results/" + "_".join([str(i) for i in instanceConfig]) + ".csv" 
            startTime = time.time()
            instance_results = self.run_instance(*instanceConfig)
            elapsedTime = time.time() - startTime
            instance_results.to_csv(filename)
            row = [instanceConfig[i] for i in range(len(instanceConfig))] + [filename, elapsedTime]
            results.append(row)   
            resultsDF = pd.DataFrame(data=results, columns=colNames)
            resultsDF.to_csv(resultsTableFilepath)
            return