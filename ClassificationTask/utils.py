import gudhi as gd
from gudhi.point_cloud.timedelay import TimeDelayEmbedding
import gudhi.representations as gdr
import numpy as np
import matplotlib.pyplot as plt
from Filtration.DTM_filtration import DTM_filtration
from Filtration.standard_filtration import standard_filtration
from ClassificationTask.generate_dataset import Dataloader




def transform_ts_to_pc(signals,delay,dim):
    """
    This function transforms a time series to a point cloud using time delay embedding.

    Parameters
    ----------
    signals : np.array
        The time series
    delay : int
        The delay for the time delay embedding
    dim : int
        The dimension for the time delay embedding
    """
    tde = TimeDelayEmbedding(dim=dim, delay=delay)
    pc = []
    for sig in signals:
        pc.append(tde(sig))
    return np.array(pc)


def transform_to_diagram(pc, DTM_bool=True,k_neighbours=5,order=2,dimension_max=2):
    """
    This function transforms a point cloud to a persistence diagram using either DTM or standard filtration.

    Parameters
    ----------
    pc : np.array
        The point cloud
    DTM_bool : bool
        If True, use DTM filtration, else use standard filtration
    k_neighbours : int
        The number of nearest neighbours
    order : int
        The order of the DTM filtration
    dimension_max : int
        The maximum dimension of the simplices to consider
    """
    diagrams = []
    if DTM_bool:
        for i in range(pc.shape[0]):
            X = pc[i,:,:].reshape((pc.shape[1],pc.shape[2]))
            p = DTM_filtration(X,k_neighbours,order,dimension_max)
            p.compute_persistence()
            D = p.persistence_intervals_in_dimension(0)
            diagrams.append(D)
    else:
        for i in range(pc.shape[0]):
            X = pc[i,:,:].reshape((pc.shape[1],pc.shape[2]))
            standard = gd.RipsComplex(points=X,max_edge_length=np.inf)
            st = standard.create_simplex_tree(max_dimension=2)
            st.compute_persistence()
            D = st.persistence_intervals_in_dimension(0)
            diagrams.append(D)
        
    return diagrams


def get_landscapes(diagram,do_plot=True):
    """
    This function computes the landscapes of a diagram.

    Parameters
    ----------
    diagram : np.array
        The persistence diagram
    do_plot : bool
        If True, plot the landscapes
    
    """
    [diagram] = gdr.DiagramSelector(use=True, point_type='finite').fit_transform([diagram])
    LS = gdr.Landscape(resolution=1000, num_landscapes=2)
    L = LS.fit_transform([diagram])

    if do_plot:
        plt.figure()
        plt.plot(L[0][:1000])
        plt.plot(L[0][1000:2000])
        #plt.plot(L[0][2000:3000])
        plt.show()

    return L


def get_silhouette(diagram, do_plot = True):
    """
    This function computes the silhouette of a diagram.

    Parameters
    ----------
    diagram : np.array
        The persistence diagram
    do_plot : bool
        If True, plot the silhouette
    
    """
    [diagram] = gdr.DiagramSelector(use=True, point_type='finite').fit_transform([diagram])
    SH = gdr.Silhouette(resolution=1000, weight=lambda x: np.power(x[1]-x[0],2))
    sh = SH.fit_transform([diagram])

    if do_plot:
        plt.figure()
        plt.plot(sh[0])
        plt.show()
    return sh

def get_images(diagram,do_plot = True):
    """
    This function computes the persistence images of a diagram.

    Parameters
    ----------
    diagram : np.array
        The persistence diagram
    do_plot : bool
        If True, plot the persistence images
    
    """
    [diagram] = gdr.DiagramSelector(use=True, point_type='finite').fit_transform([diagram])
    PI = gdr.PersistenceImage(bandwidth=5*1e-2, weight=lambda x: x[1]**0, \
                          im_range=[-.5,.5,0,.5], resolution=[100,100])
    pi = PI.fit_transform([diagram])

    if do_plot:
        plt.figure()
        plt.imshow(np.flip(np.reshape(pi[0], [100,100]), 0))
        plt.show()

    return pi


if __name__ =="__main__":

    ############## YOU CAN MODIFY THE FOLLOWING LINES ##############
    delay = 4 # Delay for the time delay embedding
    dim = 3 # Dimension for the time delay embedding
    k_neighbours = 5 # Number of nearest neighbours
    order = 3 # Order of the DTM filtration
    DTM_bool = True # If True, use DTM filtration, else use standard filtration
    ################################################################

    # Load the signals
    ts1 = Dataloader(
        dataset_path="../Datasets/ToeSegmentation1/", 
        dataset_name="ToeSegmentation1"
    )

    signals_ts1, labels_ts1 = ts1.load(split=None, return_data=True)
    signals = signals_ts1["train"][:,0:5]
    labels = labels_ts1["train"]

    # Transform the signals to point clouds
    pc = transform_ts_to_pc(signals.T, delay=delay, dim=dim)

    # Transform the point clouds to diagrams
    diagrams = transform_to_diagram(pc, DTM_bool=DTM_bool, k_neighbours=k_neighbours, order=order, dimension_max=2)

    # Compute the landscapes
    landscapes = get_landscapes(diagrams[0], do_plot=True)

    # Compute the silhouette
    silhouette = get_silhouette(diagrams[0], do_plot=True)

    # Compute the persistence images
    images = get_images(diagrams[0], do_plot=True)
