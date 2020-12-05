from basisfun import *
import random
from mpi4py import MPI


comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()

# **************** receive all parameters
Option = None
Option = comm.bcast(Option, root=0)  # Note: it has output


# **************** receive the data set as a 1d array
data = np.empty(Option['datasize'], dtype=float)
comm.Bcast([data, MPI.FLOAT], root=0)  # Note: it does not have output
data = data.reshape(Option['datashape'], order='C')  # convert to a 2d array



# initialize the output arrays
meandata = np.zeros(Option['datasize'])
numOfupdate = np.zeros(Option['NumOfAnts'], dtype=int)
numOfdim = np.zeros(Option['datashape'][1], dtype=int)



# **************** distributing the tasks
learningrate = Option['etta'] / np.sqrt(Option['Radius'])
# idx = range(rank, Option['NumOfAnts'], size)
# newdata, num_upd, num_dim = onecolony(data, Option['initialpos'][idx], Option['Radius'][idx],
#                                       Option['KNNThreshold'], Option['NumOfSteps'],
#                                       Option['HighestWeight'], Option['NumOfUpdatePerStep'],
#                                       learningrate[idx], Option['tau'])

for colony in range(rank, Option['NumOfConlonies'], size):
    idx = range(colony, Option['NumOfAnts'], Option['NumOfConlonies'])# Option['NumOfAntsPerColony'])
    newdata, num_upd, num_dim = onecolony(data, Option['initialpos'][idx], Option['Radius'][idx],
                                          Option['KNNThreshold'], Option['NumOfSteps'],
                                          Option['HighestWeight'], Option['NumOfUpdatePerStep'],
                                          learningrate[idx], Option['tau'])
    meandata += newdata.reshape((1, -1))[0]
    numOfdim += num_dim
    numOfupdate[idx] += num_upd.T[0]


# *************** initialize the output arrays
# numOfupdate = np.zeros(Option['NumOfAnts'], dtype=int)
# numOfdim = np.zeros(Option['datashape'][1], dtype=int)

# meandata = newdata.reshape((1, -1))[0]   # convert a 2d array to a 1d array
# numOfupdate[idx] += num_upd.T[0]
# numOfdim += num_dim

# *************** sending the result to the root
comm.Reduce(meandata, None, op=MPI.SUM, root=0)
comm.Reduce(numOfupdate, None, op=MPI.SUM, root=0)
comm.Reduce(numOfdim, None, op=MPI.SUM, root=0)

# computation finished
comm.Disconnect()
