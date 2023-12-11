# Laptop price predicion


## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py



MLFLOW_TRACKING_URI=https://dagshub.com/kharramahendra/Laptop-Price---End-to-End-ML-Project.mlflow \
MLFLOW_TRACKING_USERNAME=kharramahendra \
MLFLOW_TRACKING_PASSWORD=25511b164852982b155d2aeec465a612062ab5cf \
python script.py


<!-- for bash -->
export MLFLOW_TRACKING_URI=https://dagshub.com/kharramahendra/Laptop-Price---End-to-End-ML-Project.mlflow

export MLFLOW_TRACKING_USERNAME=kharramahendra 

export MLFLOW_TRACKING_PASSWORD=25511b164852982b155d2aeec465a612062ab5cf


<!-- for windows  -->
set MLFLOW_TRACKING_URI=https://dagshub.com/kharramahendra/Laptop-Price---End-to-End-ML-Project.mlflow

set MLFLOW_TRACKING_USERNAME=kharramahendra 

set MLFLOW_TRACKING_PASSWORD=25511b164852982b155d2aeec465a612062ab5cf



<!-- for notebook -->
os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/kharramahendra/Laptop-Price---End-to-End-ML-Project.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="kharramahendra"
os.environ["MLFLOW_TRACKING_PASSWORD"]="25511b164852982b155d2aeec465a612062ab5cf"