# GMDA_project

This projet has been made by Grégoire Gissot and Samy Jallouli for the course of Geometric Methods for Data Analysis at CentraleSupelec. It deals about the DTM (Distance To Measure) filtration and time series classification using topological techniques mainly based on persistence.

For the two first folder, some parameters can be modified by users and it is clearly mentionned in the different files.

The project contains three different parts/folders :
- **Filtration** : it contains three files which enable you to generate toy dataset (*generate_toy_dataset.py*) and do some standard filtrations (*standard_filtration.py*) and DTM filtrations (*DTM_filtration.py*) on these generated datasets. You can run all the files simply with the following command in the terminal "python -m Filtration.<filename>"
- **Classification Task** : it contains six files, an helper file to load a dataset ToeSegmentation1 *generate_dataset.py* (available here : http://www.timeseriesclassification.com/description.php?Dataset=ToeSegmentation1 ), some utils functions to get some topological tools to do the classification *utils.py*, one file to visualize the signals and the topological tools *vizu_representation.py*, one to create the scikit-learn pipeline used for the classification problem *create_model.py*, one to do finetuning of Time Delay Embedding parameters *TDE_finetuning.py* and the last one that do the whole training and testing of the pipeline *classify.py*. As before, you can run all the files simply with the following command in the terminal "python -m ClassificationTask.<filename>"
- **Stability** :