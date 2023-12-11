import pandas as pd
import os
from mlProject import logger
from sklearn.neighbors import KNeighborsRegressor
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self,x_train,x_test,y_train,y_test):
        knn_final = KNeighborsRegressor(algorithm=self.config.algorithm,n_neighbors=self.config.n_neighbors,weights=self.config.weights)
        knn_final.fit(x_train,y_train)

        joblib.dump(knn_final, os.path.join(self.config.root_dir, self.config.model_name))

