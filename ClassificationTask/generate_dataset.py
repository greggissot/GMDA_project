import os
import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt

class Dataloader():
    
    def __init__(
            self, 
            dataset_path: str, 
            dataset_name: str = None
        ) -> None:

        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist.")
        self.dataset_path = dataset_path
        self.dataset_name = "ArticularyWordRecognition" if dataset_name is  None else dataset_name
        
        self.dimension = None
        self.splits = None
        self.data_filename = None
        self.signals = None
        self.labels = None

    def load(
            self,  
            split: str = None, 
            return_data: bool = False
        ) -> None | tuple[np.ndarray, np.ndarray]:
        """ Load the corresponding set of signals. """

        

        if split is not None and not split in ["train", "test"]:
            raise ValueError(f"Split {split} not in ['train', 'test'].")
        self.splits = [split] if split is not None else ["train", "test"]

        self.signals = {}
        self.labels = {}
        for s in self.splits:
            # define filename and path
            data_filename = f"{self.dataset_name}_{s.upper()}.arff"
            data_file_path = os.path.join(self.dataset_path, data_filename)
            # load data
            data = pd.DataFrame(arff.loadarff(data_file_path)[0]).to_numpy(dtype=np.float32)
            
            # store data
            self.signals[s] = data[:, :-1].copy().T
            self.labels[s] = data[:, -1].copy()
            print(f"{s.upper()} signals shape: {self.signals[s].shape}")
            print(f"{s.upper()} labels shape: {self.labels[s].shape}")

        if return_data:
            return self.signals, self.labels
        

def plot_classes(signals, labels, n_samples=3,start_label=0):
    """ Plot n_samples signals for each class. """
    n_classes = len(np.unique(labels))
    fig, axs = plt.subplots(n_classes, n_samples, figsize=(20, 10))
    for c in range(n_classes):
        class_signals = signals[:,labels == c+start_label]
        for i in range(n_samples):
            axs[c, i].plot(class_signals[:,i])
            axs[c, i].set_title(f"Class {c}")
    plt.tight_layout()
    plt.show()


        

if __name__ == "__main__":
    ts1 = Dataloader(
        dataset_path="../Datasets/ToeSegmentation1/", 
        dataset_name="ToeSegmentation1"
    )

    signals_ts1, labels_ts1 = ts1.load(split=None, return_data=True)

    plot_classes(signals_ts1["train"], labels_ts1["train"], n_samples=3)