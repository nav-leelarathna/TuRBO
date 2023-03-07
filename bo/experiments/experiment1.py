import sys
sys.path.append("../")
from bo import TurboM, Turbo1

'''
Things to change
trust_regions
batch_size 
dimensionality of problem
seed

We want to evaluate whether using trust regions does truly give us a benefit compared to normal BO, and also in setting where total number of runs is equal, i.e. increasing batch size.

Secondly, I hypothesize that the benefit gained by the bandits diminished quickly as the dimensionality of the problem increases, test by ablating over the dimensionality of the problem.

Replicate Figure 7 - shows why single agent BO only explores and optimises within a highly local volume of the search space

Replicate Figure 7 - showing that increasing batch size gives us a linear improvement, plotting reward vs steps
'''
