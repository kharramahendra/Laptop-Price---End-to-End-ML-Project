import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    # I have done all advanced pre processing techniques in data validation step and validated data for next step
    # here I am going to split the data and finally transform using columntransformer


    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        # Split the data into training and test sets.
        X = data.drop('price',axis=1)
        y = data['price']
        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        logger.info("Splited data into training and test sets")
        # logger.info(x_train.shape)
        # logger.info(x_test.shape)
        print("----------------------------------------------")
        print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
        return x_train,x_test,y_train,y_test
    
    def transformation(self,x_train,x_test,y_train,y_test):
        
        cat_cols = ['brand', 'OS', 'gpu_type', 'processor_brand', 'processor_version']
        cat_transformer = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(sparse=False,drop='first',handle_unknown='ignore')),
                ("scaler", StandardScaler()),
                ]
                )
        num_cols = [ 'spec_rating', 'Ram', 'ROM', 'ROM_type', 'display_size',
                    'resolution_width', 'resolution_height', 'warranty', 'cpu_core',
                    'cpu_threads', 'processor_gen']
        num_cat_transformer = Pipeline(
            steps=[
                # ("encoder", LabelEncoder()),
                ("scaler", StandardScaler()),
            ]
        )

        transformer = ColumnTransformer(
            transformers=[
                ('categorical_transformer',cat_transformer,cat_cols),
                ("numerical_transformer",num_cat_transformer,num_cols)
                ]
                )
        final_x_train = transformer.fit_transform(x_train)
        logger.info(f'training data transformed, shape:{final_x_train.shape}')
        final_x_test = transformer.transform(x_test)
        logger.info(f'testing data transformed, shape:{final_x_test.shape}')

        import joblib
        joblib.dump(transformer, self.config.transformer_path)
        logger.info("Transformer dumped at given location")


        return final_x_train,final_x_test,y_train,y_test
        