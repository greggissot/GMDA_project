import numpy as np
import sklearn.neighbors       as skn
from ClassificationTask.generate_dataset import Dataloader
from ClassificationTask.utils import transform_ts_to_pc


def mutual_information(signal,tau,nb_bins):
    """
    This function computes the mutual information between a signal and its retarded version.

    Parameters
    ----------
    signal : np.array
        The signal
    tau : int
        The delay for the retarded signal
    nb_bins : int
        The number of bins for the histogram
    """
    mini = np.min(signal)
    maxi = np.max(signal)
    p = np.zeros((nb_bins,nb_bins))
    for i in range(nb_bins):
        # Number of points of the signal in bin i
        p[i,i] = np.mean((signal >= mini + i*(maxi-mini)/nb_bins) & (signal < mini + (i+1)*(maxi-mini)/nb_bins))
        for j in range(i+1,nb_bins):
            # Number of points of the signal in bin i while the retarded signal is in bin j
            p[i,j] = np.mean((signal >= mini + i*(maxi-mini)/nb_bins) & (signal < mini + (i+1)*(maxi-mini)/nb_bins) & (np.roll(signal,tau) >= mini + j*(maxi-mini)/nb_bins) & (np.roll(signal,tau) < mini + (j+1)*(maxi-mini)/nb_bins))

    mutual_information = 0

    for i in range(nb_bins):
        for j in range(nb_bins):
            if p[i,j] != 0:
                mutual_information += p[i,j]*np.log(p[i,j]/(p[i,i]*p[j,j]))
    return mutual_information


def find_optimal_tau(signal,nb_bins,min_tau=1,max_tau=10):
    """
    This function finds the optimal delay for the retarded signal for only one signal.

    Parameters
    ----------
    signal : np.array
        The signal
    nb_bins : int
        The number of bins for the histogram
    min_tau : int
        The minimum delay
    max_tau : int
        The maximum delay
    """
    mutual_informations = []
    for tau in range(min_tau,max_tau):
        mutual_informations.append(mutual_information(signal,tau,nb_bins))
    return np.argmax(mutual_informations)+min_tau

def search_tau(signals,nb_bins,min_tau=1,max_tau=10):
    """
    This function searches the optimal delay for the retarded signals as a majority vote.

    Parameters
    ----------
    signals : np.array
        The signals
    nb_bins : int
        The number of bins for the histogram
    min_tau : int
        The minimum delay
    max_tau : int
        The maximum delay
    """
    taus = np.zeros(max_tau)
    for i in range(signals.shape[0]):
        tau = find_optimal_tau(signals[i,:],nb_bins,min_tau,max_tau)
        taus[tau]+=1
    # Return the key with maximal values
    return np.argmax(taus)

def false_nearest_neighbours(signal,dim,threshold,tau):
    """
    This function computes the number of false nearest neighbours for a signal in the TDE embedding of a proposed dim.

    Parameters
    ----------
    signal : np.array
        The signal
    dim : int
        The dimension for the time delay embedding
    threshold : float
        The threshold for the false nearest neighbours
    tau : int
        The delay for the time delay embedding
    """

    X_embedded = transform_ts_to_pc([signal], delay=tau, dim=dim)[0]

    neighbor = \
        skn.NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X_embedded)
    distances, indices = neighbor.kneighbors(X_embedded)
    distance = distances[:, 1]
    X_first_nbhrs = signal[indices[:, 1]]

    epsilon = 2. * np.std(signal)
    tolerance = 10

    neg_dim_delay = - dim * tau
    distance_slice = distance[:neg_dim_delay]
    X_rolled = np.roll(signal, neg_dim_delay)
    X_rolled_slice = slice(len(signal) - len(X_embedded), neg_dim_delay)
    X_first_nbhrs_rolled = np.roll(X_first_nbhrs, neg_dim_delay)

    neighbor_abs_diff = np.abs(
        X_rolled[X_rolled_slice] - X_first_nbhrs_rolled[:neg_dim_delay]
        )

    false_neighbor_ratio = np.divide(
        neighbor_abs_diff, distance_slice,
        out=np.zeros_like(neighbor_abs_diff, dtype=float),
        where=(distance_slice != 0)
        )
    false_neighbor_criteria = false_neighbor_ratio > tolerance

    limited_dataset_criteria = distance_slice < epsilon

    n_false_neighbors = \
        np.sum(false_neighbor_criteria * limited_dataset_criteria)
    return n_false_neighbors


def find_optimal_dim(signal,tau,threshold,dim_min=2,dim_max=10):
    """
    This function finds the optimal dimension for the time delay embedding for only one signal.

    Parameters
    ----------
    signal : np.array
        The signal
    tau : int
        The delay for the retarded signal
    threshold : float
        The threshold for the false nearest neighbours
    dim_min : int
        The minimum dimension
    dim_max : int
        The maximum dimension
    """
    false_neighbours = []
    for dim in range(dim_min,dim_max):
        false_neighbours.append(false_nearest_neighbours(signal,dim,threshold,tau))
    return np.argmax(false_neighbours)+dim_min

def search_dim(signals,tau,threshold,dim_min=2,dim_max=10):
    """
    This function searches the optimal dimension for the time delay embedding as a majority vote.

    Parameters
    ----------
    signals : np.array
        The signals
    tau : int
        The delay for the retarded signals
    threshold : float
        The threshold for the false nearest neighbours
    dim_min : int
        The minimum dimension
    dim_max : int
        The maximum dimension
    """
    dims = np.zeros(dim_max)
    dims[0:dim_min] = [np.inf]*dim_min
    for i in range(signals.shape[0]):
        dim = find_optimal_dim(signals[i,:],tau,threshold,dim_min,dim_max)
        dims[dim]+=1
    # Return the key with maximal values
    return np.argmin(dims)





if __name__ == "__main__":

    ########### YOU CAN MODIFY THESE LINES ############
    min_tau = 1
    max_tau = 5
    nb_bins = 20
    dim_min = 2
    dim_max = 5
    ###################################################

    ts1 = Dataloader(
        dataset_path="../Datasets/ToeSegmentation1/", 
        dataset_name="ToeSegmentation1"
    )

    signals_ts1, labels_ts1 = ts1.load(split=None, return_data=True)
    signals = signals_ts1["train"][:,:]



    tau = search_tau(signals.T,20,min_tau,max_tau)
    dim = search_dim(signals_ts1['train'].T,tau,0.5,dim_min,dim_max)

    print("Delay = ", tau)
    print("Dimension = ", dim)
