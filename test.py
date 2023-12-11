from mlProject.pipeline.prediction import PredictionPipeline
# import pandas as pd
# from pathlib import Path
# df = pd.read_csv(Path('artifacts/data_validation/preprocessed_data.csv'))
# data = df.head(1)

data = ['HP', 73.0, 8, 512, 1, 15.6, 1920.0, 1080.0,
       'Windows 11 OS', 1, 'Radeon', 6, 12, 'AMD', 5, '5']
pr = PredictionPipeline()


print(int(pr.predict(data)[0]))