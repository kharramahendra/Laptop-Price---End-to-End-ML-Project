from mlProject.components.data_validation import DataValiadtion
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger
from mlProject.components.data_transformation import DataTransformation
from pathlib import Path

STAGE_NAME = "Data transformation for model training "

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                x_train,x_test,y_train,y_test = data_transformation.train_test_spliting()
                return data_transformation.transformation(x_train,x_test,y_train,y_test)
            else:
                raise Exception("You data has not valid shcema or Some error in pre processing.")
        except Exception as e:
            print(e)
            
            



    
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e