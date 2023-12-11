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



MLFLOW_TRACKING_URI="your project uri" \
MLFLOW_TRACKING_USERNAME="your name" \
MLFLOW_TRACKING_PASSWORD="your password" \
python script.py


<!-- for bash -->
export MLFLOW_TRACKING_URI=project uri

export MLFLOW_TRACKING_USERNAME=your name 

export MLFLOW_TRACKING_PASSWORD=password


<!-- for windows  -->
set MLFLOW_TRACKING_URI=project uri

set MLFLOW_TRACKING_USERNAME=your name 

set MLFLOW_TRACKING_PASSWORD=password



<!-- for notebook -->
os.environ["MLFLOW_TRACKING_URI"]="uri"
os.environ["MLFLOW_TRACKING_USERNAME"]="your name"
os.environ["MLFLOW_TRACKING_PASSWORD"]="your password"