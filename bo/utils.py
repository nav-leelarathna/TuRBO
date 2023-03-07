import random
import numpy as np
from bo.functions import Levy, RobotPush, RoverControl
from bo.models import ModelFactory

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

        
def getFunc(func_name):
    match func_name:
        case "Levy":
            return Levy(10)
        case "RoverControl":
            return RoverControl()
        case "RobotPush":
            return RobotPush()