import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neighbors import NearestNeighbors
from numpy import linalg as LA
import time
import scipy.io as sio
import random


def LocalPCA(X):
    D = X.shape[1]
    pca = PCA(n_components=D)
    pca.fit(X)
    EigVal = pca.singular_values_
    EigVal = EigVal / np.sum(EigVal)
    EigVec = pca.components_

    S = np.append([EigVal[:D-1] - EigVal[1:]], [EigVal[D-1]])
    # for i in range(S.shape[0]):
    #     S[i] = (i + 1) * S[i]
    dim = np.argmax(S) + 1

    Error = np.sum(EigVal[dim:])
    return EigVal, EigVec, dim, Error


def InitialPosition(data, radius, numofant):
    """
    It distribute ants in the data space
    :param data: the data set
    :param radius: the radius of the balls which divide the data space
    :param numofant: number of ants who walk in the data space
    :return:
    """
    l = list(range(data.shape[0]))
    init = []
    t = 0
    while l and (t < numofant):
        i_c = random.choice(l)
        init.append(i_c)
        _, idx = NeighborhoodRadius(data, i_c, radius)
        for i in idx:
            if i in l:
                l.remove(i)
        t += 1
    if t < numofant:
        morePos = list(np.random.randint(0, data.shape[0], numofant - t))
        for p in morePos:
            init.append(p)
    if len(init) != numofant:
        print('Error: number of ant is not equal to the number of initialized places')
    idx_init = np.random.permutation(init)

    return idx_init


def RandomSelection(Weight, NumOfUpdPerStep):
    """
    It is to select randomly the next destination and the points for moving
    :param Weight: how close samples are to the tangent space
    :param NumOfUpdPerStep: number of neighbors to move
    :return: it return the indices for the next step and to move
    """
    NumUpdatePerStep = np.copy(NumOfUpdPerStep)
    if len(Weight) < NumUpdatePerStep:
        NumUpdatePerStep = len(Weight)
        # ******** Transition probability for the next step
    P = Weight / np.sum(Weight)
    CSP = np.cumsum(P)
    r = np.random.rand(1)
    NextPoint = np.argwhere(CSP >= r).reshape((1,-1))[0,0]

    # ********** Selection of a point for attraction
    Weight[Weight == 1] = 0.01
    P = (1 - Weight) / np.sum(1 - Weight)
    CSP = np.cumsum(P)
    r = np.random.rand(NumUpdatePerStep)
    AbsorptionPoint = np.zeros(NumUpdatePerStep, dtype=int)
    for i in range(NumUpdatePerStep):
        AbsorptionPoint[i] = np.argwhere(CSP >= r[i])[0]

    return NextPoint, AbsorptionPoint


def NeighborhoodRadius(Data, CurrentPos, Radius):
    if (Data.shape[1] < 20):
        neigh = NearestNeighbors(radius=Radius, algorithm='kd_tree').fit(Data)
    else:
        neigh = NearestNeighbors(radius=Radius, algorithm='ball_tree').fit(Data)

    Dd = Data[CurrentPos].reshape(1, -1)
    indicesNN = neigh.radius_neighbors(Dd, return_distance=False)[0]
    Neighbors = Data[indicesNN]
    return Neighbors, indicesNN


def NeighborhoodKNN(Data, CurrentPos, K):

    if (Data.shape[1] < 20):
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='kd_tree').fit(Data)  # fast for low dimensional data
    else:
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(Data)  # fast for high-dim data
        # nbrs = NearestNeighbors(n_neighbors=K, algorithm='brute').fit(Data)
    distancesNN, indicesNN = nbrs.kneighbors([Data[CurrentPos]])
    indicesNN = indicesNN[0]
    Neighbor = Data[indicesNN]
    return Neighbor, indicesNN


def MBSMWeight(X, U, p):
    D = X.shape[1]
    M = np.mean(X, axis=0)
    Distance = LA.norm(np.dot(X - M, np.identity(D) - np.dot(U.T, U)), axis=1)[:, None].T[0]
    a = np.percentile(Distance, p)
    Weight = np.heaviside(a - Distance, 0) * (1 - Distance / a)
    Weight[Weight < 0] = 0
    return Weight


def Attraction_MBMS(X, Mean, EigVec):
    D = X.shape[1]
    Delta = np.dot(Mean - X, np.identity(D) - np.dot(EigVec.T, EigVec))
    return Delta

# def SoftWeight(NeighborsVector,EigValue,EigVector) :
#    Cosine = np.abs(NeighborsVector * EigVector);
#    Cosine(isnan(Cosine)) = 0 ;
#    Weight = (Cosine/np.sum(Cosine,2)) * EigValue ;
#    return Weight


def Plots(X_o, X, color, i):
    D = X_o.shape[1]

    Mini = X_o.min(axis=0).reshape((D, 1))
    Maxi = X_o.max(axis=0).reshape((D, 1))
    MinMax = np.concatenate((Mini, Maxi), axis=1)

    if D==2:
        figure, axes = plt.subplots(figsize=(8, 8))#nrows=1, ncols=2, figsize=(15,8))
        axes.scatter(X_o[:, 0], X_o[:, 1], s=40, c='g', marker=".")
        axes.scatter(X[:, 0], X[:, 1], s=40, c='r', marker=".")

    else:
        fig = plt.figure(figsize=(14, 16))
        gs1 = gridspec.GridSpec(2, 2)
        gs1.update(wspace=0.01, hspace=0.01)  # set the spacing between axes.
        plt.suptitle(f"Manifold Learning in %{i}-th iteration", fontsize=14)

        ax = fig.add_subplot(gs1[0], projection='3d')
        ax.scatter(X_o[:, 0], X_o[:, 1], X_o[:, 2], c=color, s=20, cmap=plt.get_cmap('hsv'), marker=".")
        plt.xlim(MinMax[0, 0], MinMax[0, 1]), plt.ylim(MinMax[1, 0], MinMax[1, 1]), ax.set_zlim(MinMax[2, 0], MinMax[2, 1])
        # ax.set_title("Noisy data set")
        # ax.axis('equal')
        ax.view_init(25, -120)

        ax = fig.add_subplot(gs1[2], projection='3d')
        ax.scatter(X_o[:, 0], X_o[:, 1], X_o[:, 2], c=color, s=20, cmap=plt.get_cmap('hsv'), marker=".")
        plt.xlim(MinMax[0, 0], MinMax[0, 1]), plt.ylim(MinMax[1, 0], MinMax[1, 1]), ax.set_zlim(MinMax[2, 0], MinMax[2, 1])
        # ax.set_title("Noisy data set")
        ax.view_init(0, -90)

        ax = fig.add_subplot(gs1[1], projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, s=20, cmap=plt.get_cmap('hsv'), marker=".")
        plt.xlim(MinMax[0, 0], MinMax[0, 1]), plt.ylim(MinMax[1, 0], MinMax[1, 1]), ax.set_zlim(MinMax[2, 0], MinMax[2, 1])
        # ax.set_title("Denoised data set")
        ax.view_init(25, -120)

        ax = fig.add_subplot(gs1[3], projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, s=20, cmap=plt.get_cmap('hsv'), marker=".")
        plt.xlim(MinMax[0, 0], MinMax[0, 1]), plt.ylim(MinMax[1, 0], MinMax[1, 1]), ax.set_zlim(MinMax[2, 0], MinMax[2, 1])
        # ax.set_title("Denoised data set")
        ax.view_init(0, -90)
    plt.show()
    time.sleep(1)


def Extract_d_r(StrategyNumber, n_k):
    d = StrategyNumber // n_k + 1
    r_i = StrategyNumber % n_k
    return d, r_i


def Generate_r(antstrategy, numofstrategies, MinRadius, MaxRadius):  # check????
    Edges = np.linspace(MinRadius, MaxRadius, numofstrategies + 1)  # .reshape((1,-1)); #check : +1
    # d, i_r = Extract_d_r(StrategyPerAnt, n_k)
    Radius = np.zeros(len(antstrategy))  # .reshape((1,-1)) ;
    for i in range(numofstrategies):
        Ind = np.argwhere(antstrategy == i).T[0]
        if np.size(Ind) != 0:
            Radius[Ind] = Edges[i] + (Edges[i + 1] - Edges[i]) * np.random.rand(len(Ind))  # .reshape((1,-1)) ;
    return Radius


def SelectStrategies(NumOfPopu, Frequency):
    # AntStrategy (n_ant * 1): which strategy is selected for each ant,
    # PopulationOfStrategies (n_st * 1): Number of people per strategy
    Bins = np.append(0, np.cumsum(Frequency))
    Random = np.sort(np.random.rand(NumOfPopu))
    PopulationOfStrategies, a = np.histogram(Random, bins=Bins)
    AntStrategy = np.zeros(NumOfPopu, dtype=int)  # .reshape((1,-1)) ;
    S = 0
    for i in range(len(PopulationOfStrategies)):
        NumOfAntPerSpecies = PopulationOfStrategies[i]
        AntStrategy[S:S + NumOfAntPerSpecies] = i * np.ones(NumOfAntPerSpecies)  # .reshape((1,-1)) ;
        S = S + NumOfAntPerSpecies

    return AntStrategy, PopulationOfStrategies


def Replicator(Frequency, AntStrategy, NumOfUpdatesPerAnt, NumOfSteps):
    # Use replicator to return new frequencies of strategies
    # StrategiesPerAnt saves the strategy of each ant
    # NumOfUpdatesPerAnt save the number of update that each ant did
    Fitness = np.zeros(len(Frequency))  # .reshape((1,-1)) ; # Fitness for each strategy
    for i in range(len(Frequency)):
        if np.sum(AntStrategy == i) == 0:
            pass
        else:
            Fitness[i] = np.mean(NumOfUpdatesPerAnt[AntStrategy == i]) / NumOfSteps

    Frequency = Frequency * (Fitness - Frequency * Fitness) + Frequency
    if np.sum(Frequency < 0) != 0:
        print('A frequency is negative')
    if np.sum(Frequency) == 0:
        print('all Frequencies are zero')
    Frequency = Frequency / np.sum(Frequency)
    return Frequency, Fitness

def ant_walk(data, radius, k, numofsteps, HighestWeight, NumOfUpdPerStep, learning_rate, error_thr):
    """
    It perform the algorithm for an individual ant
    :param Data: data set, nearest neighbors, distances
    :param learning_rate: parameters of the algorithm
    :param error_thr: it saves the output on a file
    :param radius: 1: if you want to save the result
    :param radius: 1: if you want to save the result
    :return: it returns the pheromone distribution
    """
    # *************** Initialization
    Data = np.copy(data)

    numofupdate = 0
    numOfvisiteddim = np.zeros(Data.shape[1], dtype=int)  # .reshape((1,-1))
    CurrentPos = np.random.randint(0, Data.shape[0] - 1, 1)[0]

    # ************** Performing Ant colony
    for Loop in range(numofsteps):
        # ******** Extracting Neighbors *********
        Neighbors, Index = NeighborhoodRadius(Data, CurrentPos, radius)
        if len(Neighbors) < k:
            Neighbors, Index = NeighborhoodKNN(Data, CurrentPos, k)

        # ************** Local PCA **************
        EigVal, EigVec, dim, Error = LocalPCA(Neighbors)
        Mean = np.mean(Neighbors, axis=0)

        # ******* Computing weight values *******
        Weight = MBSMWeight(Neighbors, EigVec[0:dim], HighestWeight)

        # ******* random selection of the next point
        NextPoint, AbsorptionPoint = RandomSelection(Weight, NumOfUpdPerStep)
        CurrentPos = Index[NextPoint]

        # ************** Attraction *************
        if Error > error_thr:
            numOfvisiteddim[dim - 1] += 1
            numofupdate += 1

            # ****** Updating rule : MBMS ******
            Mean = np.mean(Neighbors, axis=0)
            DeltaData = learning_rate * Attraction_MBMS(Data[Index[AbsorptionPoint]],
                                                        Mean, EigVec[0:dim])
            Data[Index[AbsorptionPoint]] = Data[Index[AbsorptionPoint]] + DeltaData

    return Data, numofupdate, numOfvisiteddim


def onecolony(data, initialpos, radius, k, numofsteps, HighestWeight, NumOfUpdPerStep, learning_rate, error_thr):
    """
    It perform the algorithm for a colony (a set of ant) simultaneously
    :param Data: data set, nearest neighbors, distances
    :param initialpoint: a vector containing the starting point for each ant
    :param numofsteps: number of steps that each ant walks
    :param radius: an array which specifies the neighborhood size for each ant
    :param k: minimum number of points that an ant can observe
    :param HighestWeight: it controls how much far an ant can go from the manifold
    :param NumOfUpdPerStep: number of samples that an ant can change in each step
    :param learning_rate: controls the change
    :param error_thr: it saves the output on a file
    :return: it returns the updated (denoised) data set
    """
    # *************** Initialization
    Data = np.copy(data)
    numofants = len(radius)

    numofupdate = np.zeros((numofants, 1), dtype=int)  # 0
    numOfvisiteddim = np.zeros(Data.shape[1], dtype=int)  # .reshape((1,-1))
    CurrentPos = initialpos  #np.random.randint(0, Data.shape[0] - 1, 1)[0]

    # ************** Performing Ant colony
    for Loop in range(numofsteps):
        for ant in range(numofants):
            # ******** Extracting Neighbors *********
            Neighbors, Index = NeighborhoodRadius(Data, CurrentPos[ant], radius[ant])
            if len(Neighbors) < k:
                Neighbors, Index = NeighborhoodKNN(Data, CurrentPos[ant], k)

            # ************** Local PCA **************
            EigVal, EigVec, dim, Error = LocalPCA(Neighbors)
            Mean = np.mean(Neighbors, axis=0)

            # ******* Computing weight values *******
            Weight = MBSMWeight(Neighbors, EigVec[0:dim], HighestWeight)

            # ******* random selection of the next point
            NextPoint, AbsorptionPoint = RandomSelection(Weight, NumOfUpdPerStep)
            CurrentPos[ant] = Index[NextPoint]

            # ************** Attraction *************
            if Error > error_thr:
                numOfvisiteddim[dim - 1] += 1
                numofupdate[ant] += 1

                # ****** Updating rule : MBMS ******
                Mean = np.mean(Neighbors, axis=0)
                DeltaData = learning_rate[ant] * Attraction_MBMS(Data[Index[AbsorptionPoint]],
                                                            Mean, EigVec[0:dim])
                Data[Index[AbsorptionPoint]] = Data[Index[AbsorptionPoint]] + DeltaData

    return Data, numofupdate, numOfvisiteddim


def ComputeDistToGroundTruth(X_c, X):
    return np.mean(LA.norm(X_c - X, axis=1))


def generate_spherecircle():
    N1 = 60
    N2 = 1000
    radius = 6
    theta = np.linspace(0, 2 * np.pi, N1)
    phi = np.linspace(0, np.pi / 2, N1)
    theta, phi = np.meshgrid(theta, phi)

    x = (radius * np.sin(phi) * np.cos(theta)).reshape((-1, 1))
    y = (radius * np.sin(phi) * np.sin(theta)).reshape((-1, 1))
    z = (radius * np.cos(phi)).reshape((-1, 1))
    data_clean_1 = np.concatenate((x, y, z), axis=1)

    t = np.pi * np.random.rand(N2, 1)
    data_clean_2 = radius * np.concatenate((np.cos(t), np.zeros((N2, 1)), -np.sin(t)), axis=1)

    data_clean = np.concatenate((data_clean_1, data_clean_2), axis=0)

    labels = 0.01 + np.concatenate((np.abs(2 - 1 * theta.reshape(-1, 1)), 4 - 2 * np.abs(np.pi / 2 - t)), axis=0)
    # sio.savemat('data.mat', {'data_clean': data_clean})
    # Plots(data_clean,data_clean, labels[0],0)
    return data_clean, labels

# data_clean, _ = generate_spherecircle()
# data = data_clean + 0.3 * np.random.randn(data_clean.shape[0], data_clean.shape[1])

def generate_spiral(N):
    t = 3 + 12 * np.random.rand(N, 1)
    data_clean = np.concatenate((0.04 * t * np.sin(t), 0.04 * t * np.cos(t)), axis=1)

    return data_clean

def OneCircle_varNoise():
    N1 = 1700
    N2 = 300

    # t1 = 2 * np.pi * np.random.rand(N1, 1)
    t1 = np.concatenate((np.pi/2 + 1.5 * np.pi * np.random.rand(N1, 1), 2 * np.pi + np.pi / 2 * np.random.rand(N2, 1)))
    data_clean_1 = 4 * np.concatenate((np.cos(t1), np.sin(t1) - 1), axis=1)
    data_1 = data_clean_1 + 0.3 * np.concatenate((np.cos(2 * (t1 - np.pi / 2) / np.pi),
                                                  np.cos(2 * (t1 - np.pi / 2) / np.pi)),
                                                 axis=1) * np.random.randn(N1+N2, 2)

    # t2 = np.concatenate((np.pi + np.pi * np.random.rand(N2, 1),
    #                      np.pi / 2 + np.pi / 2 * np.random.rand(int(N2 / 8), 1)), axis=0)
    # t2 = np.concatenate((np.pi + 1.5 * np.pi * np.random.rand(N2, 1),
    #                      np.pi / 2 + np.pi / 2 * np.random.rand(int(N2 / 8), 1)), axis=0)
    # data_clean_2 = 2.5 * np.concatenate((np.cos(t2), np.sin(t2)+1), axis=1)
    # data_2 = data_clean_2 + 0.2 * np.random.randn(len(t2), 2)

    # data_clean = np.concatenate((data_clean_1,data_clean_2), axis=0)
    # data = np.concatenate((data_1, data_2), axis=0)
    Plots(data_clean_1, data_1, 'g', 0)

    return data_clean_1, data_1


# OneCircle_varNoise()

def TwoCircle_varNoise():
    N1 = 2000
    N2 = 1200
    t1 = 2 * np.pi * np.random.rand(N1, 1)
    data_clean_1 = 2.5 * np.concatenate((np.cos(t1), np.sin(t1)-1), axis=1)
    data_1 = data_clean_1 + 0.3 * np.concatenate((np.cos(2*(t1 - np.pi / 2) / np.pi),
                                                  np.cos(2*(t1 - np.pi / 2) / np.pi)),
                                                  axis=1) * np.random.randn(N1, 2)

    # t2 = np.concatenate((np.pi + np.pi * np.random.rand(N2, 1),
    #                      np.pi / 2 + np.pi / 2 * np.random.rand(int(N2 / 8), 1)), axis=0)
    t2 = np.concatenate((np.pi + 1.5 * np.pi * np.random.rand(N2, 1),
                         np.pi / 2 + np.pi / 2 * np.random.rand(int(N2 / 8), 1)), axis=0)
    data_clean_2 = 2.5 * np.concatenate((np.cos(t2), np.sin(t2)+1), axis=1)
    data_2 = data_clean_2 + 0.2 * np.random.randn(len(t2), 2)

    data_clean = np.concatenate((data_clean_1,data_clean_2), axis=0)
    data = np.concatenate((data_1, data_2), axis=0)
    # Plots(data_clean, data, 'g', 0)

    return data_clean, data

# TwoCircle_varNoise()
# [d,l] = generate_spherecircle()
# l=l.reshape((1,4600))
# print(l.shape)
# Plots(d,d,l,0)