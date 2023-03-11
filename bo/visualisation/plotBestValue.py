import pandas as pd
import os 
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

path_to_data = "../data/Levy_10/sweep_results/0_Levy_10_hesbo_1000_10_5.csv"
resultsFile = "../data/Levy_10/results.csv"
def getSweepData(resultsFile):
    with open(resultsFile, "r") as rf:
        df = pd.read_csv(rf)
        filename = df["datapath"]
        sign = df["function_sign"]
        computeTime = df["compute_time"]
        model = df["model"]
        function = df["function"]
        modelMaximising = df["is_model_maximising"]
        return model,function,computeTime, sign, modelMaximising, filename

def summaryStats(resultsFile):
    sweepData = getSweepData(resultsFile)
    

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
    plt.plot(sign * np.fmax.accumulate(fX), 'r', lw=1)  # Plot cumulative minimum as a red line
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