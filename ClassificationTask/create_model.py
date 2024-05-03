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


def create_model(pipe, params, cv):
    """
    Create a model
    Parameters
    ----------
    pipe : Pipeline
        The pipeline
    params : dict
        The parameters
    cv : int
        The number of folds
    Returns
    -------
    GridSearchCV
        The model
    """
    model = skm.GridSearchCV(pipe, params, cv=cv)
    return model

if __name__ == "__main__":
    
    
    pipe = skl.Pipeline([("Separator", gdr.DiagramSelector(limit=np.inf, point_type="finite")),
                     ("Scaler",    gdr.DiagramScaler(scalers=[([0,1], skp.MinMaxScaler())])),
                     ("TDA",       gdr.PersistenceImage()),
                     ("Estimator", sks.SVC())])
    
    param =    [{"Separator__use":      [True],
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
             "Estimator":           [sks.SVC(kernel="rbf", gamma="auto")]},

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
    
    model =  create_model(pipe, param, 3)
