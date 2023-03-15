
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
        return [(s,f,m, me, bs, n, noise) for f in configs["function"] for m in configs["model"] for s in configs["seed"] for me in configs["max_evals"] for bs in configs["batch_size"]for n in configs["n_init"] for noise in configs["noise"]]
    
    
    def get_instance(self, seed, func, model, max_evals, batch_size, n_init, noise):
        set_seed(seed)
        func = getFunc(func)
        # if noise > 0:
        #     noistFunc = NoisyFunc(func)
        space = func.getParameterSpace()
        lb = np.array([p.min for p in space.parameters])
        ub = np.array([p.max for p in space.parameters])
        factory = ModelFactory(func, lb, ub, n_init)
        if func.dim > 80:
            low_dim = 10 
        else:
            low_dim = 4
        # match model:
        if model== "gp":
            model = factory.getGP_BO(batch_size, max_evals)
        elif model== "turbo1":
            model = factory.getTurbo1(batch_size, max_evals)
        elif model[:5]=="turbo":
            trustRegions = int(model[5:])
            model = factory.getTurboM(batch_size=batch_size, max_evals=max_evals, trust_regions=trustRegions)
        elif model== "hesbo":
            model = factory.getHesbo(low_dim=low_dim, max_evals=max_evals)
        elif model== "nelder-mead":
            model = factory.getNelderMead(max_evals=max_evals)
        elif model== "bfgs":
            model = factory.getBFGS(max_evals=max_evals)
        elif model== "cobyla":
            model = factory.getCOBYLA(max_evals=max_evals)
        # elif model== "bobyqua":
        #     model = factory.getBOBYQUA(max_evals=max_evals)
        elif model== "cmaes":
            model = factory.getCMAES(max_evals=max_evals, batch_size=batch_size, sigma=1)
        elif model== "random":
            model = factory.getRandom(max_evals=max_evals)
        else:
            raise Exception(f"Unknown model: {model}")
        if func.maximising == model.maximising:
            func.setSign(1)
        else:
            func.setSign(-1)
        return model, func
    
    def optimize(self, model, func):
        model.optimize()
        X = model.X
        fX = model.fX
        numEvaluations = np.array(list(range(X.size)))
        return pd.DataFrame(list(zip(numEvaluations,X,fX)), columns=["evals","X", "fX"]), func.sign, model.maximising
    
    def mock_results_file(self):
        name = self.sweepConfig["name"]
        datapath = f"data/{name}/sweep_results"
        os.makedirs(datapath, exist_ok=True)

        colNames = list(self.sweepConfig["configurations"].keys()) + ["datapath", "compute_time", "function_sign", "is_model_maximising"]
        results = []
        resultsTableFilepath = f"data/{name}/resultsMock.csv"
        instanceConfigs = self.generate_instances()
        for instanceConfig in instanceConfigs:
            filename = f"data/{name}/sweep_results/" + "_".join([str(i) for i in instanceConfig]) + ".csv" 
            elapsedTime = -1
            model, func = self.get_instance(*instanceConfig)
            functionSign = func.sign 
            modelMaximising = model.maximising
            row = [instanceConfig[i] for i in range(len(instanceConfig))] + [filename, elapsedTime, functionSign,modelMaximising]
            assert len(row) == len(colNames)
            results.append(row)   
        resultsDF = pd.DataFrame(data=results, columns=colNames)
        resultsDF.to_csv(resultsTableFilepath)

    def run(self):
        name = self.sweepConfig["name"]
        datapath = f"data/{name}/sweep_results"
        os.makedirs(datapath, exist_ok=True)

        colNames = list(self.sweepConfig["configurations"].keys()) + ["datapath", "compute_time", "function_sign", "is_model_maximising"]
        results = []
        instanceConfigs = self.generate_instances()
        # random.shuffle(instanceConfigs)

        resultsTableFilepath = f"data/{name}/results2.csv"
        print(f"Will write results to {resultsTableFilepath}")
        for j, instanceConfig in enumerate(instanceConfigs):
            # maybe add functionality to skip if instance already contained in results.csv so that we can restart 
            print(f"Starting run {j+1}/{len(instanceConfigs)}")
            print(instanceConfig)
            filename = f"data/{name}/sweep_results/" + "_".join([str(i) for i in instanceConfig]) + ".csv" 
            startTime = time.time()
            # instance_results,functionSign, modelMaximising = self.run_instance(*instanceConfig)
            model, func = self.get_instance(*instanceConfig)
            instance_results,functionSign, modelMaximising = self.optimize(model, func)
            elapsedTime = time.time() - startTime
            instance_results.to_csv(filename)
            row = [instanceConfig[i] for i in range(len(instanceConfig))] + [filename, elapsedTime, functionSign,modelMaximising]
            assert len(row) == len(colNames)
            results.append(row)   
            resultsDF = pd.DataFrame(data=results, columns=colNames)
            resultsDF.to_csv(resultsTableFilepath)
            # return