import argparse
from basisfun import *
from scipy.io import loadmat
import scipy.io as sio
from mpi4py import MPI
import distutils.spawn # added *******************************
import sys
from sklearn import datasets
import time
import os

def parse_args():
    '''
    Parses arguments.
    '''
    parser = argparse.ArgumentParser(description="Run EM3A.")

    parser.add_argument('--input', nargs='?', default='sshape',
                        help='Input data set')

    parser.add_argument('--outputdir', nargs='?', default='run1',
                        help='denoised manifold')

    parser.add_argument('--tau', type=float, default=0.01,
                        help='the threshold for error value. Default is 0.01.')

    parser.add_argument('--etta', type=float, default=0.1,
                      help='learning rate')

    parser.add_argument('--Ncarrying', type=int, default=1,
                      help='number of points that an agent moves in every steps. Default is 1.')

    parser.add_argument('--rinit', type=float, default=0.5,
                      help='the minimum distance between initial position of agents. Default is 0.5.')

    parser.add_argument('--workers', type=int, default=3,
                        help='Number of parallel workers. Default is 5.')

    # argument about agents
    parser.add_argument('--Nsteps', type=int, default=100,
                        help='Number of steps for an agent. Default is 100.')


    parser.add_argument('--Nagentspercolony', type=int, default=10,
                        help='number of agents in a colony. Default is size of data set / num Of steps.')

    parser.add_argument('--Ncolonies', type=int, default=10,
                        help='Number of colonies. Default is 10.')
    
    parser.add_argument('--minNeighbors', type=int, default=-1,
                        help='minimum number of neighbors. Default is the number of feature.')

    parser.add_argument('--minradius', type=float, default=0.0,
                        help='minimum of neighborhood size (r). Default is 0.1.')

    parser.add_argument('--maxradius', type=float, default=2.0,
                        help='maximum of neighborhood size (r). Default is 1.')

    parser.add_argument('--Nstrategies', type=int, default=10,
                        help='number of strategies. Default is 10.')

    parser.add_argument('--Niter', type=int, default=300,
                        help='number of iteration. Default is 50.')

    parser.add_argument('--plot', type=int, default=100,
                        help='ploting results after some iterations. Default is 10.')

    return parser.parse_args()

def argsTodict():
    Option = dict()

    # ******** Hyper-para: Manifold shape ********
    Option['tau'] = args.tau  # Thresholding for error value
    Option['etta'] = args.etta  # Learning Rate
    Option['NumOfUpdatePerStep'] = args.Ncarrying  # Number of points an ant can move
    Option['radius_division_dataspace'] = args.rinit  # the radius for dividing the data space
    
    # ********* Hyper-para: Random walk **********
    # Option['WeightMethod'] = 'MBSM'
    Option['HighestWeight'] = 50  # percentile of weights
    Option['NumOfColonies'] = args.Ncolonies
    Option['NumOfAntsPerColony'] = args.Nagentspercolony
    Option['NumOfAnts'] = Option['NumOfColonies'] * Option['NumOfAntsPerColony']  # Population of ants
    Option['NumOfSteps'] = args.Nsteps
    Option['KNNThreshold'] = args.minNeighbors


    # ********* Hyper-para.: strategies **********
    Option['MinRadius'] = args.minradius
    Option['MaxRadius'] = args.maxradius
    Option['NumOfStrategies'] = args.Nstrategies
    Option['FreqOfStrategies'] = np.ones(Option['NumOfStrategies']) / Option['NumOfStrategies']

    # ************* stop criteria ****************    
    Option['maxiter'] = args.Niter
    Option['workers'] = args.workers
    Option['nplot'] = args.plot # plotting after some iterations
    inputFile = args.input
    dir_output = args.outputdir

    return Option, inputFile, dir_output


def main():
    # *******************************************
    # **************** Loading Data *************
    # *******************************************
    Option, inputFile, outputdir = argsTodict() 
    dir_input = os.path.join('./data', inputFile+'.mat')
    dir_output = os.path.join('./output', outputdir)
    os.mkdir(dir_output)

    f = sio.loadmat(dir_input)
    data = f['data']
    labels = 'g'

    # *********************************************
    # ******** Fixing hyper-parameters ************
    # *********************************************    
    # Option['KNNThreshold'] = 10#'default' if Option['KNNThreshold']>data.shape[1] else data.shape[1]
    if Option['KNNThreshold']<data.shape[1]:
        Option['KNNThreshold']= data.shape[1]+1
    data_origin = np.copy(data)
    Option['datashape'] = data.shape
    Option['datasize'] = data.size
    D = data.shape[1]
    maxofprocess = Option['workers']

    PopulationOverTime = []
    FitnessOverTime = []
    NumOfDim = []
    
    start_time = time.time()
    for i in range(Option['maxiter']):  
        print("********************************************")      
        print(f"{i}-th iteration")                

        # Initial position and strategies of ants
        print(f"Initialize positions and strategies of {Option['NumOfAnts']} ants in {Option['NumOfColonies']} colonies!")

        Option['initialpos'] = InitialPosition(data, Option['radius_division_dataspace'], Option['NumOfAnts'])
        Option['AntStrategy'], PopulationOfStrategies = SelectStrategies(Option['NumOfAnts'],
                                                                         Option['FreqOfStrategies'])
        Option['Radius'] = Generate_r(Option['AntStrategy'], Option['NumOfStrategies'],
                                      Option['MinRadius'], Option['MaxRadius'])

        # *********************************************** #
        # ********** Start of parallelization *********** #
        # *********************************************** #        
        print("ants start walking! please wait... :)")
        # mpi_info = MPI.Info.Create()
        # mpi_info.Set("add-hostfile", "slurm.hosts")
        # mpi_info.Set("host", "slurm.hosts")
        comm = MPI.COMM_SELF.Spawn(sys.executable,
                                   args=['Colonies.py'], 
                                   maxprocs=maxofprocess)#.Merge()
                                   
        comm.bcast(Option, root=MPI.ROOT)
        comm.Bcast([data.reshape((1, -1))[0], MPI.FLOAT], root=MPI.ROOT)

        meandata = np.empty(data.size, dtype=float)
        numOfdim = np.empty(data.shape[1], dtype=int)
        numOfupdate = np.empty(Option['NumOfAnts'], dtype=int)

        # receives the updates
        comm.Reduce(None, meandata, op=MPI.SUM, root=MPI.ROOT)
        comm.Reduce(None, numOfupdate, op=MPI.SUM, root=MPI.ROOT)
        comm.Reduce(None, numOfdim, op=MPI.SUM, root=MPI.ROOT)
        data = meandata.reshape(data.shape) / Option['NumOfColonies']

        comm.Disconnect()
        # *********************************************** #
        # *********** End of parallelization ************ #
        # *********************************************** #

        print(f'num of update: {np.sum(numOfupdate)}')
        
        print(numOfdim)
        NumOfDim.append(list(numOfdim))

        Option['FreqOfStrategies'], Fitness = Replicator(Option['FreqOfStrategies'], Option['AntStrategy'],
                                                         numOfupdate, Option['NumOfSteps'])

        PopulationOverTime.append(list(PopulationOfStrategies))
        FitnessOverTime.append(list(Fitness))
        print(f'Population: {PopulationOfStrategies}')
        print("fitness: {}\n".format(np.round(Fitness,2)))

        # ************** Save and plot *****************
        if i % 1 == 0:
            FileName_mat = os.path.join(dir_output, inputFile + str(i) + '.mat')
            # FileName_py = os.path.join(dir_output, inputFile + str(i))
            Times = time.time() - start_time
            print('time: ', Times)
            sio.savemat(FileName_mat, {'data_origin': data_origin, 'data': data,
                                       'Time': Times, 'labels': labels, 'Option': Option,
                                       'PopulationOverTime': PopulationOverTime,
                                       'FitnessOverTime': FitnessOverTime, 'NumOfDim': NumOfDim})
            # np.savez(FileName_py, data_origin=data_origin, data=data,
            #          Time=Times, labels=labels, Option=Option,
            #          PopulationOverTime=PopulationOverTime,
            #          FitnessOverTime=FitnessOverTime, NumOfDim=NumOfDim)

        if i % Option['nplot'] == 0:
            Plots(data_origin, data, 'g', i)


if __name__=="__main__":
    args = parse_args()
    main()
