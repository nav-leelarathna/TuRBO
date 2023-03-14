import random
import numpy as np
from bo.functions import Levy, RobotPush, RoverControl, Ackley
from bo.models import ModelFactory

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

        
def getFunc(func_name):
    if func_name[:4] == "Levy":
        dims = int(func_name.split("_")[-1])
        return Levy(dims)
    # if func_name ==  "Levy_10":
    #     return Levy(10)
    # elif func_name == "Levy_100":
    #     return Levy(100)
    elif func_name[:6] == "Ackley":
        dims = int(func_name.split("_")[-1])
        return Ackley(dims)
    # elif func_name ==  "Ackley_10":
    #     return Ackley(10)
    # elif func_name ==  "Ackley_100":
    #     return Ackley(100)
    elif func_name ==  "RoverControl":
        return RoverControl()
    elif func_name ==  "RobotPush":
        return RobotPush()
    else:
        raise Exception(f"Unknown func: {func_name}")
    
    