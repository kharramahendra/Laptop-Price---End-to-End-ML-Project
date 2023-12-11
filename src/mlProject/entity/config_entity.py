from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict
    preprocessed_data:Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    transformer_path:Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_name: str
    algorithm: str
    n_neighbors: float
    weights: str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    transformer_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str