# f1_predictor/mlflow_integration.py - MLflow integration for model tracking

import os
import logging
from typing import Dict, Any, Optional
import mlflow
import mlflow.sklearn
from datetime import datetime

from .config_manager import get_config

logger = logging.getLogger(__name__)

class F1MLflowTracker:
    """MLflow integration for F1 prediction model tracking."""
    
    def __init__(self):
        """Initialize MLflow tracking."""
        self._setup_mlflow()
        self.experiment_id = None
        
    def _setup_mlflow(self):
        """Setup MLflow configuration."""
        try:
            # Get MLflow config
            mlflow_config = get_config('mlflow', {})
            
            # Set tracking URI
            tracking_uri = mlflow_config.get('tracking_uri', 'file:./mlruns')
            mlflow.set_tracking_uri(tracking_uri)
            
            # Set experiment
            experiment_name = mlflow_config.get('experiment_name', 'f1_prediction_v3')
            
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment:
                    self.experiment_id = experiment.experiment_id
                else:
                    self.experiment_id = mlflow.create_experiment(experiment_name)
            except Exception as e:
                logger.warning(f"Could not setup experiment: {e}")
                self.experiment_id = mlflow.create_experiment(experiment_name)
            
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow setup complete: {tracking_uri}, experiment: {experiment_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
    
    def start_run(self, run_name: Optional[str] = None) -> str:
        """Start a new MLflow run."""
        try:
            if run_name is None:
                run_name = f"f1_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id)
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Started MLflow run: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            return ""
    
    def log_model_training(self, 
                          model: Any,
                          model_metadata: Any,
                          feature_names: list,
                          hyperparameters: Dict[str, Any],
                          performance_metrics: Dict[str, float]):
        """Log model training details to MLflow."""
        try:
            # Log hyperparameters
            mlflow.log_params(hyperparameters)
            
            # Log metrics
            mlflow.log_metrics(performance_metrics)
            
            # Log model metadata
            if hasattr(model_metadata, 'model_type'):
                mlflow.log_param("model_type", model_metadata.model_type)
            if hasattr(model_metadata, 'algorithm'):
                mlflow.log_param("algorithm", model_metadata.algorithm)
            mlflow.log_param("feature_count", len(feature_names))
            if hasattr(model_metadata, 'feature_hash'):
                mlflow.log_param("feature_hash", model_metadata.feature_hash)
            if hasattr(model_metadata, 'version'):
                mlflow.log_param("version", model_metadata.version)
            
            # Log model artifact
            if get_config('mlflow.log_models', True):
                model_type = getattr(model_metadata, 'model_type', 'unknown')
                algorithm = getattr(model_metadata, 'algorithm', 'unknown')
                mlflow.sklearn.log_model(
                    model, 
                    f"{model_type}_{algorithm}",
                    registered_model_name=f"f1_{model_type}_{algorithm}"
                )
            
            # Log feature names as artifact
            if get_config('mlflow.log_artifacts', True):
                with open("feature_names.txt", "w") as f:
                    f.write("\n".join(feature_names))
                mlflow.log_artifact("feature_names.txt")
                os.remove("feature_names.txt")
            
            logger.info("Model training logged to MLflow successfully")
            
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}")
    
    def log_prediction_metrics(self, prediction_metrics: Dict[str, Any]):
        """Log prediction performance metrics."""
        try:
            # Log prediction-specific metrics
            for key, value in prediction_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"prediction_{key}", value)
                else:
                    mlflow.log_param(f"prediction_{key}", str(value))
            
            logger.debug("Prediction metrics logged to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log prediction metrics: {e}")
    
    def end_run(self):
        """End the current MLflow run."""
        try:
            mlflow.end_run()
            logger.debug("MLflow run ended")
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")
    
    def load_model(self, model_name: str, version: Optional[str] = None) -> Optional[Any]:
        """Load a registered model from MLflow."""
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model from MLflow: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {e}")
            return None
    
    def search_runs(self, filter_string: str = "", max_results: int = 10) -> list:
        """Search for runs in the experiment."""
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                max_results=max_results,
                order_by=["start_time DESC"]
            )
            return runs.to_dict('records') if not runs.empty else []
            
        except Exception as e:
            logger.error(f"Failed to search runs: {e}")
            return []

# Global MLflow tracker instance
_mlflow_tracker = None

def get_mlflow_tracker() -> F1MLflowTracker:
    """Get global MLflow tracker instance."""
    global _mlflow_tracker
    if _mlflow_tracker is None:
        _mlflow_tracker = F1MLflowTracker()
    return _mlflow_tracker

def log_model_training(model: Any, metadata: Any, features: list, 
                      params: Dict[str, Any], metrics: Dict[str, float]):
    """Log model training using global tracker."""
    return get_mlflow_tracker().log_model_training(model, metadata, features, params, metrics)

def start_mlflow_run(run_name: Optional[str] = None) -> str:
    """Start MLflow run using global tracker."""
    return get_mlflow_tracker().start_run(run_name)

def end_mlflow_run():
    """End MLflow run using global tracker."""
    get_mlflow_tracker().end_run() 