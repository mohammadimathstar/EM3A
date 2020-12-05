from basisfun import *
from scipy.io import loadmat
import scipy.io as sio
from mpi4py import MPI
import sys
from sklearn import datasets
import time


# **********************************************
# **************** Generating Data *************
# **********************************************
Option = dict()
Option['NoiseLevel'] = 0.15
Option['NumOfMdf'] = 4000
D = 3
FileName = 'test'

# f = np.load('Multi_0.npz')
# data = f['data']
# labels = f['labels']
# data_clean = f['data_clean']
# data_origin = f['data_origin']

# Option=f['Option']
# PopulationOverTime=f['PopulationOverTime']
# FitnessOverTime=f['FitnessOverTime']
# NumOfDim=f['NumOfDim']

# ************** 3d data sets *************
data_clean, labels = datasets.make_s_curve(Option['NumOfMdf'], 0, random_state=None)
# data_clean, labels = datasets.make_swiss_roll(n_samples=Option['NumOfMdf'], noise=0.0, random_state=None)
# if D > 3:
#     data_clean = np.concatenate((data_clean, np.zeros((data_clean.shape[0], D - 3))), axis=1)
data = data_clean + Option['NoiseLevel'] * np.random.randn(data_clean.shape[0], data_clean.shape[1])

# ************* 2d data sets **************
# data_clean, labels = datasets.make_moons(n_samples=Option['NumOfMdf'], noise=.05)   # noisy two moon
# data_clean, labels = datasets.make_circles(n_samples=Option['NumOfMdf'], factor=.5, noise=.05)

# data_clean, labels = generate_spherecircle()
# data_clean = generate_spiral(Option['NumOfMdf'])
#
# data = data_clean + Option['NoiseLevel'] * np.random.randn(data_clean.shape[0], data_clean.shape[1])
# data_clean, data = TwoCircle_varNoise()
# data_clean, data = OneCircle_varNoise()

# labels = 'g'
# Plots(data_clean, data, 'g',0)


# Mini = data.min(axis=0).reshape((D, 1))
# Maxi = data.max(axis=0).reshape((D, 1))
# if Mini[2] == Maxi[2]:
#     Maxi[2] += 0.5
# MinMax = np.concatenate((Mini, Maxi), axis=1)

# **********************************************
# ************* Fixing parameters **************
# **********************************************
data_origin = np.copy(data)
Option['datashape'] = data.shape
Option['datasize'] = data.size
maxofprocess = 5

# ******** Hyper-para: Manifold shape ********
Option['tau'] = 0.01  # Thresholding for error value
Option['etta'] = 0.1  # Learning Rate
Option['NumOfUpdatePerStep'] = 5  # Number of points an ant can move
Option['radius_division_dataspace'] = 0.5  # the radius for dividing the data space

# ********* Hyper-para: Random walk **********
Option['WeightMethod'] = 'MBSM'
Option['HighestWeight'] = 50  # percentile of weights
Option['NumOfConlonies'] = 20
Option['NumOfAntsPerColony'] = 10  # number of ants in a colony
Option['NumOfAnts'] = Option['NumOfConlonies'] * Option['NumOfAntsPerColony']  # Population of ants
Option['NumOfSteps'] = 100  # Number of steps per iteration per ant
Option['KNNThreshold'] = D  # Least number of neighbors

# ********* Hyper-para.: strategies **********
Option['MinRadius'] = 0  # Minimum radius (view range)
Option['MaxRadius'] = 1.5  # Maximum radius (view range)
Option['NumOfStrategies'] = 10
Option['FreqOfStrategies'] = np.ones(Option['NumOfStrategies']) / Option['NumOfStrategies']

# ************* stop criteria ****************
Option['StopCriteria'] = 0.1
Option['maxiter'] = 50

PopulationOverTime = []
FitnessOverTime = []
NumOfDim = []
dist_vec = []
start_time = time.time()
for i in range(Option['maxiter']):
    print(data.shape)
    print(f"{i}-th iteration")
    # d = ComputeDistToGroundTruth(data_clean, data)
    # dist_vec.append(d)
    # print(f"distance to the graound truth is: {d}")

    # ************** Save and plot *****************
    if i % 100 == 0:
        FileName_mat = FileName + str(i) + '.mat'
        FileName_py = FileName + str(i)
        Times = time.time() - start_time
        print('time: ', Times)
        sio.savemat(FileName_mat, {'data_origin': data_origin, 'data': data,'data_clean': data_clean,
                                   'Time': Times, 'labels': labels, 'Option': Option,
                                   'dist_vec': dist_vec, 'PopulationOverTime': PopulationOverTime,
                                   'FitnessOverTime': FitnessOverTime, 'NumOfDim': NumOfDim})
        np.savez(FileName_py, data_clean=data_clean, data_origin=data_origin, data=data,
                 Time=Times, labels=labels, Option=Option,
                 dist_vec=dist_vec, PopulationOverTime=PopulationOverTime,
                 FitnessOverTime=FitnessOverTime, NumOfDim=NumOfDim)

    if i % 1 == 0:
        Plots(data_origin, data, 'g', i)

    # Initial position for ants
    Option['initialpos'] = InitialPosition(data, Option['radius_division_dataspace'], Option['NumOfAnts'])

    # To select the strategies (neighborhood size)
    Option['AntStrategy'], PopulationOfStrategies = SelectStrategies(Option['NumOfAnts'],
                                                                     Option['FreqOfStrategies'])
    Option['Radius'] = Generate_r(Option['AntStrategy'], Option['NumOfStrategies'],
                                  Option['MinRadius'], Option['MaxRadius'])

    # *********************************************** #
    # ********** Start of parallelization *********** #
    # *********************************************** #
    comm = MPI.COMM_SELF.Spawn(sys.executable,
                               args=['Colonies.py'], #['Colony.py'],
                               maxprocs=maxofprocess)

    comm.bcast(Option, root=MPI.ROOT)
    comm.Bcast([data.reshape((1, -1))[0], MPI.FLOAT], root=MPI.ROOT)

    meandata = np.empty(data.size, dtype=float)
    numOfdim = np.empty(data.shape[1], dtype=int)
    numOfupdate = np.empty(Option['NumOfAnts'], dtype=int)

    # receives the updates
    comm.Reduce(None, meandata, op=MPI.SUM, root=MPI.ROOT)
    comm.Reduce(None, numOfupdate, op=MPI.SUM, root=MPI.ROOT)
    comm.Reduce(None, numOfdim, op=MPI.SUM, root=MPI.ROOT)
    data = meandata.reshape(data.shape) / Option['NumOfConlonies']


    comm.Disconnect()
    # *********************************************** #
    # *********** End of parallelization ************ #
    # *********************************************** #

    print(f'num of update: {np.sum(numOfupdate)}')
    # TODO: maybe you need to adapt error_thr according to its radius
    # Option['Error'] = Option['tau'] * Option['Radius'] * np.ones(len(Option['Radius']))

    # TODO: maybe you need to limit the amount of learning rate when etta is small.
    # learning_rate = Option['etta'] / np.sqrt(Option['Radius'][Ant])

    print(numOfdim)
    NumOfDim.append(list(numOfdim))

    Option['FreqOfStrategies'], Fitness = Replicator(Option['FreqOfStrategies'], Option['AntStrategy'],
                                                     numOfupdate, Option['NumOfSteps'])

    PopulationOverTime.append(list(PopulationOfStrategies))
    FitnessOverTime.append(list(Fitness))
    print(f'Population: {PopulationOfStrategies}')
    print(f'fitness: {Fitness}\n')


    # if max(Fitness) < Option['StopCriteria'] :
    #   print('Finish')
    #  break;



end_time = time.time()
duration = end_time - start_time
print(duration)

Plots(data_origin, data, 'g', 0)
sio.savemat(FileName+str(maxofprocess)+'.mat', {'data_origin': data_origin, 'data': data,
                                                'data_clean': data_clean, 'labels': labels,
                                                'Option': Option, 'PopulationOverTime': PopulationOverTime,
                                                'FitnessOverTime': FitnessOverTime, 'NumOfDim': NumOfDim,
                                                'duration': duration, 'maxofprocess': maxofprocess})
np.savez(FileName+str(maxofprocess), data_origin=data_origin, data=data,
         data_clean=data_clean, labels=labels, Option=Option,
         PopulationOverTime=PopulationOverTime, FitnessOverTime=FitnessOverTime,
         NumOfDim=NumOfDim, duration=duration, maxofprocess=maxofprocess)

