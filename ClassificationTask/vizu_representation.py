import matplotlib.pyplot as plt
import numpy as np
from ClassificationTask.generate_dataset import Dataloader
from ClassificationTask.utils import transform_ts_to_pc,transform_to_diagram,get_silhouette,get_images




def transform_ts_to_vizu(ts,get_func,DTM_bool=True,k_neighbours=5,order=2,dimension_max=2,delay=1,dim=4):
    """ Transform a time series to a visualization (silhouette or image).
     
    Parameters
    ----------
    ts : np.array
        The time series
    get_func : function
        The function to apply to the time series (get_silhouette or get_images)
    DTM_bool : bool
        If True, use DTM filtration, else use standard filtration
    k_neighbours : int
        The number of nearest neighbours
    order : int
        The order of the DTM filtration
    dimension_max : int
        The maximum dimension of the simplices to consider
    delay : int
        The delay for the time delay embedding
    dim : int
        The dimension for the time delay embedding"""
    
    pc = transform_ts_to_pc(ts,delay=delay,dim=dim)[0]
    diagrams = transform_to_diagram(pc.reshape(1,pc.shape[0],pc.shape[1]),DTM_bool,k_neighbours,order,dimension_max)[0]
    vizu = get_func(diagrams,False)
    return vizu

def plot_silhouette_per_classes(signals, labels, n_samples=3,start_label=0,DTM_bool=True,k_neighbours=5,order=2,dimension_max=2,delay=1,dim=4):
    """ Plot n_samples silhouette for each class. """
    n_classes = len(np.unique(labels))
    fig, axs = plt.subplots(n_classes, n_samples, figsize=(20, 10))
    for c in range(n_classes):
        class_signals = signals[:,labels == c+start_label]
        for i in range(n_samples):
            axs[c, i].plot(transform_ts_to_vizu([class_signals[:,i].T],get_silhouette,DTM_bool,k_neighbours,order,dimension_max,delay,dim)[0,:])
            axs[c, i].set_title(f"Class {c}")
    plt.tight_layout()
    plt.show()

def plot_image_per_classes(signals, labels, n_samples=3,start_label=0,DTM_bool=True,k_neighbours=5,order=2,dimension_max=2,delay=1,dim=4):
    """ Plot n_samples persistence image for each class. """
    n_classes = len(np.unique(labels))
    fig, axs = plt.subplots(n_classes, n_samples, figsize=(20, 10))
    for c in range(n_classes):
        class_signals = signals[:,labels == c+start_label]
        for i in range(n_samples):
            pi = transform_ts_to_vizu([class_signals[:,i].T],get_images,DTM_bool,k_neighbours,order,dimension_max,delay,dim)[0]
            axs[c, i].imshow(np.flip(np.reshape(pi, [100,100]), 0))
            axs[c, i].set_title(f"Class {c}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":  

    ts1 = Dataloader(
        dataset_path="../Datasets/ToeSegmentation1/", 
        dataset_name="ToeSegmentation1"
    )

    signals_ts1, labels_ts1 = ts1.load(split=None, return_data=True)
    signals = signals_ts1["train"]
    labels = labels_ts1["train"]

    plot_silhouette_per_classes(signals, labels, n_samples=3,start_label=0,DTM_bool=True,k_neighbours=10,order=3,dimension_max=2)
    plot_image_per_classes(signals, labels, n_samples=3,start_label=0,DTM_bool=True,k_neighbours=10,order=3,dimension_max=2)
