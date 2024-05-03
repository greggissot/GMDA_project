import gudhi.representations as gdr
import numpy as np
import sklearn.decomposition   as skd
import sklearn.manifold        as skf
import sklearn.pipeline        as skl
import sklearn.svm             as sks
import sklearn.ensemble        as ske
import sklearn.neighbors       as skn
import sklearn.model_selection as skm
import sklearn.preprocessing   as skp
from ClassificationTask.generate_dataset import Dataloader
from ClassificationTask.utils import transform_to_diagram,transform_ts_to_pc
from ClassificationTask.create_model import create_model
from ClassificationTask.TDE_finetuning import search_tau,search_dim

if __name__ == "__main__":


        ################## YOU CAN MODIFY THE FOLLOWING CODE ##################
        min_tau = 1 # Minimum tau for the embedding
        max_tau = 5 # Maximum tau for the embedding
        min_dim = 2 # Minimum dimension of the embedding
        max_dim = 5 # Maximum dimension of the embedding
        k_neighbours = 25 # Number of neighbours for the DTM
        order = 3 # Order of the DTM
        DTM_use = True # If you want to use DTM set it to True
        ########################################################################


        ts1 = Dataloader(
            dataset_path="../Datasets/ToeSegmentation1/", 
            dataset_name="ToeSegmentation1"
        )
    
        signals_ts1, labels_ts1 = ts1.load(split=None, return_data=True)
        signals_train = signals_ts1["train"][:,:]
        labels_train = labels_ts1["train"]

        signals_test = signals_ts1["test"][:,70:170]
        labels_test = labels_ts1["test"][70:170]


        print("Searching parameters for embedding")
        tau = search_tau(signals_train.T,20,min_tau,max_tau)
        dim = search_dim(signals_train.T,tau,1,min_dim, max_dim)
        print("tau = ", tau)
        print("dim = ", dim)

        print("Preprocess train data")
        pc_train = transform_ts_to_pc(signals_train.T,tau,dim)
        diagrams_train = transform_to_diagram(pc_train,DTM_bool=DTM_use,k_neighbours=k_neighbours,order=order,dimension_max=2)


        print("Preprocess test data")
        pc_test = transform_ts_to_pc(signals_test.T,tau,dim)
        diagrams_test = transform_to_diagram(pc_test,DTM_bool=DTM_use,k_neighbours=k_neighbours,order=order,dimension_max=2)

        
        pipe = skl.Pipeline([("Separator", gdr.DiagramSelector(limit=np.inf, point_type="finite")),
                        ("Scaler",    gdr.DiagramScaler(scalers=[([0,1], skp.MinMaxScaler())])),
                        ("TDA",       gdr.PersistenceImage()),
                        ("Estimator", sks.SVC())])
        
        param = [{"Separator__use":      [True],
                "Scaler__use":         [False, True],
                "TDA":                 [gdr.PersistenceImage()], 
                "TDA__bandwidth":      [0.05, 0.5],
                "TDA__resolution":     [[100,100],[50,50]],
                "Estimator":           [sks.SVC(kernel="rbf", gamma="auto")]},
                            
                {"Separator__use":      [True],
                "Scaler__use":         [False, True],
                "TDA":                 [gdr.Silhouette()], 
                "TDA__resolution":     [100,50],
                "Estimator":           [ske.RandomForestClassifier()]},
            
                {"Separator__use":      [True],
                "Scaler__use":         [False, True],
                "TDA":                 [gdr.PersistenceImage()], 
                "TDA__bandwidth":      [0.05, 0.5],
                "TDA__resolution":     [[100,100],[50,50]],
                "Estimator":           [ske.RandomForestClassifier()]},
                
                {"Separator__use":      [True],
                "Scaler__use":         [False, True],
                "TDA":                 [gdr.Silhouette()], 
                "TDA__resolution":     [100,50],
                "Estimator":           [ske.RandomForestClassifier()]},
                
                {"Separator__use":      [True],
                "Scaler__use":         [False, True],
                "TDA":                 [gdr.PersistenceImage()], 
                "TDA__bandwidth":      [0.05, 0.5],
                "TDA__resolution":     [[100,100],[50,50]],
                "Estimator":           [skn.KNeighborsClassifier()]},
                        
                {"Separator__use":      [True],
                "Scaler__use":         [False, True],
                "TDA":                 [gdr.Silhouette()], 
                "TDA__resolution":     [100,50],
                "Estimator":           [skn.KNeighborsClassifier()]},
                ]
        
        model = create_model(pipe, param, 3)
        print("Fitting model")
        model.fit(diagrams_train, labels_train)
        print(model.best_params_)

        print("Train accuracy = " + str(model.score(diagrams_train, labels_train)))
        print("Test accuracy  = " + str(model.score(diagrams_test,  labels_test)))