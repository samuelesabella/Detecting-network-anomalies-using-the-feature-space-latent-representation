# Main purpose
The project aims to study the use of the feature space latent representation for the purpose of network monitoring. We trained different machine learning models to learn a latent representation that can be used both to detect similar devices and detect anomalies. The learning and deploying process is unsupervised and can be run using raw traffic pcaps.

# Usage on new data using pretrained models
Download and unzip the [checkpoints]() and place it in a new directory called `checkpoints` in the project root. Follow the instructions in `lab/` to create the dataset from raw pcaps files and place them in `dataset/`. Open the jupyter notebook `notebooks/CICIDS2017_VisualizationTool.ipynb` to visualize the dataset. Use `CICIDS2017_T-SNE_Visualizer.ipynb` to visualize the model output

# Project structure
```
/
│   lab.requirements.txt: pip packages required to start the lab
│   requirements.txt: pip packages required to start the training/prediction 
│
└───notebooks: 
│   │   CICIDS2017_T-SNE_Visualizer.ipynb: visualize and study the model predictions
│   │   CICIDS2017_VisualizationTool.ipynb: visualizes the dataset extracted
│   │   ... (the other notebooks can be ignored and may be removed by future commits)
│   
└───folder2
    |   cicids2017.py: grid search model selection and evaluation
    │   data_generator.py: extracts the data from ntopng backend database
    │   Seq2Seq.py: class modelling sequence to sequence autoencoders
    |   AnchoredTs2Vec.py: class modelling triplet loss based models
    |   ntopng_constants.py: mainly used to set ntopng features to be used by the models
    |   Callbacks.py: training callbacks
    |   AnomalyDetector.py: defines the class used to build contexts from dataframes(WindowedDataGenerator)
                            and the base class of our models implementing point-wise application features
```
