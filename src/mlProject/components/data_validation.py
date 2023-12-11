import os
from mlProject import logger
import pandas as pd
import numpy as np
from mlProject.entity.config_entity import DataValidationConfig
from mlProject.utils.advanced_preprocessing import parse_processor_name,extract_cores_threads,get_gpu_type

class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config
    
    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.preprocessed_data)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
            
            print(validation_status)
            return validation_status
        
        except Exception as e:
            raise e


    def advanced_processing(self)-> bool:
        try:
            df = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(df.columns)
            all_schema = self.config.all_schema.keys()

            # Step 1 lets handle the processor column

            processors = list(df['processor'])
            new = []
            for processor in processors:
                 value = parse_processor_name(processor)
                 new.append(value)
            
            processor_data = []
            for obj in new:
                if obj is None:
                    processor_data.append([None,None,None,None])
                else:
                    processor_data.append([obj['company'],obj['generation'],obj['version'],obj['model_type']])
            # adding new columns (feture engineering)
            df[['processor_brand','processor_gen','processor_version','processor_model']] = processor_data


            # Step 2 handle gpu column

            gpus = list(df['GPU'])
            gpu_data = []
            for gpu in gpus:
                value = get_gpu_type(gpu)
                gpu_data.append(value)
            # adding new column
            df['gpu_type'] = gpu_data

            
            # Step 3 handling cpu column

            cpu_data = []
            for cpu in list(df['CPU']):
                cpu_data.append(extract_cores_threads(cpu))
            
            # adding new columns
            df[['cpu_core','cpu_threads']] = cpu_data


            # Remove all unwanted columns from the data
            data = df.drop(['Unnamed: 0.1', 'Unnamed: 0','name','processor','CPU','Ram_type','GPU','processor_model'],axis=1)

            # handling Ram column
            data.update(data['Ram'].apply(lambda x: int(x.split('GB')[0])))

            # handling ROM
            data.update(data['ROM'].apply(lambda x: int(x.split('GB')[0]) if 'GB' in x else int(x.split('TB')[0])*1024))

            # handling ROM_type
            data.update(data['ROM_type'].apply(lambda x: 1 if 'SSD' in x else 0))

            # handling missing values in processor_gen column
            data.update(data['processor_gen'].fillna(data['processor_gen'].mode()[0],inplace=True))

            # handling missing values in processor_brand column
            data.update(data['processor_brand'].fillna(data['processor_brand'].mode()[0],inplace=True))

            # handling missing values in processor_model which depends on processor_brand
            for brand in data['processor_brand'].value_counts().index:
                data.update(data[data['processor_brand']==brand]['processor_version'].replace(np.nan,data[data['processor_brand']==brand]['processor_version'].mode()[0]))
            
            # handling missing values in gpu_type
            data['gpu_type'].fillna(data['gpu_type'].mode()[0],inplace=True)

            # OS column have little issue
            data.update(data['OS'].replace('Windows 11  OS','Windows 11 OS'))
            data.update(data['OS'].replace('Windows 10  OS','Windows 10 OS'))


            # some os the colums have numerical values but their dtype is object so handling them
            data[['Ram','ROM','ROM_type','processor_gen']] = data[['Ram','ROM','ROM_type','processor_gen']].apply(np.int64)
            print(data.isnull().sum())
            print(data.columns)
            logger.info("Advanced pre processing is done")

            data.to_csv(self.config.preprocessed_data)

            logger.info("data file saved to given path")
            return True
        
        except Exception as e:
            raise e

