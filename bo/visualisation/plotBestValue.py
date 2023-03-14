import pandas as pd
import os 
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

path_to_data = "../data/Levy_10/sweep_results/0_Levy_10_hesbo_1000_10_5.csv"
resultsFile = "../data/Levy_10/results.csv"
resultsFile = "../data/robotPush/results.csv"
resultsFile = "../data/100D/results.csv"
def loadFile(resultsFile):
    with open(resultsFile, "r") as rf:
        df = pd.read_csv(rf)
    return df 

def getSweepData(resultsFile):
        df = loadFile(resultsFile)
        filename = df["datapath"]
        sign = df["function_sign"]
        computeTime = df["compute_time"]
        model = df["model"]
        function = df["function"]
        modelMaximising = df["is_model_maximising"]
        return model,function,computeTime, sign, modelMaximising, filename

def summaryStats(resultsFile):
    sweepData = getSweepData(resultsFile)

def meanOfSeries(series):
    means = []
    numSeries = len(series)
    length = len(series[0])
    for i in range(len(series)):
        assert length == len(series[i])
    for i in range(length):
        elementMean = 0
        for j in range(len(series)):
            elementMean += series[j][i]
        elementMean /= numSeries
        means.append(elementMean)
    return means

def aggregate(df):
    fig, ax = plt.subplots()
    models = df['model'].unique().tolist()
    maxEvals = df['max_evals'].unique().tolist()[0]
    colours = ['b', 'r', 'g', 'v']
    for k, model in enumerate(models):
        model_rows = df.loc[df["model"]==model]
        datapaths = model_rows["datapath"].tolist()
        function_signs = model_rows["function_sign"].tolist()
        models_maximising = model_rows["is_model_maximising"].tolist()
        fXAggs = []
        for i in range(len(datapaths)):
            datapath = datapaths[i]
            function_sign = function_signs[i]
            model_maximising = models_maximising[i]
            fXAgg = loadRunFile("../" + datapath, function_sign, model_maximising)
            fXAggs.append(fXAgg[:maxEvals])
        fXAggsMean = meanOfSeries(fXAggs)
        ax.plot(fXAggsMean, colours[k], lw=3, label=model)
    ax.set_xlim([0, len(fXAgg)])  
    # plt.ylim([-10, 30])
    ax.set_title("Aggregates")
    ax.grid(axis='y')
    ax.legend()
    fig.tight_layout()
    plt.show()

def loadRunFile(filepath, functionSign, modelMaximising):
    df = loadFile(filepath)
    fX = df["fX"].tolist()
    foo = []
    for fxi in fX:
        if isinstance(fxi, str):
            foo.append(float(fxi[1:-1]))
        else:
            foo.append(fxi)
    fX = foo   
    # fX = [float(f[1:-1]) for f in fX]
    if modelMaximising:
        return functionSign * np.fmax.accumulate(fX)
    else:
        return functionSign*np.minimum.accumulate(fX)


def test():
    model,function,computeTime, sign, modelMaximising ,filename = getSweepData(resultsFile)
    df = pd.read_csv("../" + filename[0])
    fX = df["fX"].tolist()
    # for f in fX:
    #     print(float(f[2:-2]))

    fX = [float(f[1:-1]) for f in fX]
    # print(fX)
    sign = sign[0]
    modelMaximising = modelMaximising[0]
    fig = plt.figure(figsize=(7, 5))
    matplotlib.rcParams.update({'font.size': 16})
    # plt.plot(sign*fX, 'b.', ms=10)  #  Plot all evaluated points as blue dots
    if modelMaximising:
        plt.plot(sign * np.fmax.accumulate(fX), 'r', lw=3)  # Plot cumulative minimum as a red line
    else:
        plt.plot(sign*np.minumum.accumulate(fX), 'r', lw=3)  # Plot cumulative minimum as a red line
    plt.xlim([0, len(fX)])  
    # plt.ylim([-10, 30])
    plt.title("10D Levy function")
    plt.legend()
    plt.tight_layout()
    plt.show()

# X = turbo.X  # Evaluated points
# fX = turbo.fX  # Observed values
# ind_best = np.argmin(fX)
# f_best, x_best = fX[ind_best], X[ind_best, :]

# print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))

# fig = plt.figure(figsize=(7, 5))
# matplotlib.rcParams.update({'font.size': 16})
# plt.plot(f.sign*fX, 'b.', ms=10)  # Plot all evaluated points as blue dots
# plt.plot(f.sign*np.minimum.accumulate(fX), 'r', lw=3)  # Plot cumulative minimum as a red line
# plt.xlim([0, len(fX)])
# plt.ylim([-10, 30])
# plt.title("10D Levy function")

# plt.tight_layout()
# plt.show()
if __name__ == "__main__":
    df = loadFile(resultsFile)
    aggregate(df)