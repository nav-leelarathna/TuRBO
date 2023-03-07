import sys
sys.path.append("../")
from bo.models import ModelFactory 
from bo.functions import RobotPush

f = RobotPush()
factory = ModelFactory(f, f.lb, f.ub, 5,None, False)
hesbo = factory.getHesbo(4, 25)
hesbo.optimize()

X = hesbo.X 
fX = hesbo.fX 

print(fX)