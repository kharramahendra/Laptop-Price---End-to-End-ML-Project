import joblib 
import numpy as np
import pandas as pd
from pathlib import Path



class PredictionPipeline:
    def __init__(self):
        self.transfomer = joblib.load(Path('artifacts/data_transformation/transformer.pkl'))
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.columns = ['brand', 'spec_rating', 'Ram', 'ROM', 'ROM_type',
                              'display_size', 'resolution_width', 'resolution_height',
                              'OS', 'warranty', 'gpu_type', 'cpu_core', 'cpu_threads',
                              'processor_brand', 'processor_gen', 'processor_version']
    
    
    def predict(self, data):
        df = pd.DataFrame([data],columns=self.columns)
        transfomed_data = self.transfomer.transform(df)
        prediction = self.model.predict(transfomed_data)

        return prediction