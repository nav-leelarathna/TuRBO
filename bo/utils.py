import random
import numpy as np
from bo.functions import Levy, RobotPush, RoverControl, Ackley
from bo.models import ModelFactory

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

        
def getFunc(func_name):
    match func_name:
        case "Levy_10":
            return Levy(10)
        case "Levy_100":
            return Levy(100)
        case "Ackley_10":
            return Ackley(10)
        case "Ackley_100":
            return Ackley(100)
        case "RoverControl":
            return RoverControl()
        case "RobotPush":
            return RobotPush()