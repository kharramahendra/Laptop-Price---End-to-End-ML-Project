from mlProject.components.model_trainer import ModelTrainer
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger

STAGE_NAME = "Model Training"

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def main(self,x_train,x_test,y_train,y_test):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        trainer = ModelTrainer(config=model_trainer_config)
        print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
        trainer.train(x_train,x_test,y_train,y_test)



    
# if __name__ == '__main__':
#     try:
#         logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#         obj = ModelTrainerPipeline()
#         obj.main(x_train,x_test,y_train,y_test)
#         logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
#     except Exception as e:
#         logger.exception(e)
#         raise e