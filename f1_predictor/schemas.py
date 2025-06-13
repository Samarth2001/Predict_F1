# f1_predictor/schemas.py - Type-safe schemas for F1 prediction system

from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import hashlib

class RaceWeekendData(BaseModel):
    """Schema for race weekend data."""
    year: int = Field(..., ge=2018, le=2030)
    race_num: int = Field(..., ge=1, le=24)
    circuit: str = Field(..., min_length=1)
    race_name: str = Field(..., min_length=1)
    date: datetime
    drivers: List[str] = Field(..., min_items=10, max_items=24)
    teams: List[str] = Field(..., min_items=5, max_items=12)
    
    model_config = {"arbitrary_types_allowed": True}

class FeatureMatrix(BaseModel):
    """Schema for engineered feature matrix."""
    features: Dict[str, Any] = Field(..., description="Feature values")
    feature_names: List[str] = Field(..., min_items=1)
    target_column: Optional[str] = None
    shape: tuple = Field(..., description="Data shape (rows, cols)")
    hash: str = Field(..., description="Feature hash for version control")
    
    @field_validator('hash')
    @classmethod
    def validate_hash(cls, v):
        if len(v) != 64:  # SHA-256 hash length
            raise ValueError("Hash must be 64 characters (SHA-256)")
        return v
    
    @classmethod
    def from_dataframe(cls, df, target_col: Optional[str] = None):
        """Create FeatureMatrix from pandas DataFrame."""
        feature_names = df.columns.tolist()
        if target_col and target_col in feature_names:
            feature_names.remove(target_col)
        
        # Create feature hash for versioning
        feature_str = '|'.join(sorted(feature_names))
        feature_hash = hashlib.sha256(feature_str.encode()).hexdigest()
        
        return cls(
            features=df.to_dict('records'),
            feature_names=feature_names,
            target_column=target_col,
            shape=df.shape,
            hash=feature_hash
        )
    
    def to_dataframe(self):
        """Convert back to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.features)

class ModelMetadata(BaseModel):
    """Schema for model metadata."""
    model_type: str = Field(..., description="Type of model (qualifying/race)")
    algorithm: str = Field(..., description="ML algorithm used")
    feature_hash: str = Field(..., description="Hash of features used")
    feature_names: List[str] = Field(..., min_items=1)
    training_date: datetime = Field(default_factory=datetime.now)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    imputation_values: Dict[str, float] = Field(default_factory=dict)
    version: str = Field(default="3.0.0")

class PredictionResult(BaseModel):
    """Schema for prediction results."""
    driver: str = Field(..., min_length=1)
    team: str = Field(..., min_length=1)
    predicted_position: float = Field(..., ge=1, le=24)
    predicted_rank: int = Field(..., ge=1, le=24)
    confidence_score: float = Field(..., ge=0, le=1)
    model_type: str = Field(..., description="qualifying or race")
    prediction_timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = {"validate_assignment": True}

class WeekendPredictionResults(BaseModel):
    """Schema for complete weekend prediction results."""
    race_weekend: RaceWeekendData
    qualifying_predictions: Optional[List[PredictionResult]] = None
    pre_quali_race_predictions: Optional[List[PredictionResult]] = None
    post_quali_race_predictions: Optional[List[PredictionResult]] = None
    weekend_summary: Dict[str, Any] = Field(default_factory=dict)
    prediction_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('qualifying_predictions', 'pre_quali_race_predictions', 'post_quali_race_predictions')
    @classmethod
    def validate_predictions(cls, v):
        if v and len(v) > 24:
            raise ValueError("Cannot have more than 24 drivers")
        return v

class SessionData(BaseModel):
    """Schema for session data from FastF1."""
    session_type: str = Field(..., pattern="^(qualifying|race)$")
    year: int = Field(..., ge=2018)
    round_number: int = Field(..., ge=1)
    circuit: str = Field(..., min_length=1)
    session_date: datetime
    weather_data: Optional[Dict[str, Any]] = None
    results: List[Dict[str, Any]] = Field(..., min_items=1)
    
    model_config = {"arbitrary_types_allowed": True}

class ConfigurationSchema(BaseModel):
    """Schema for system configuration."""
    model_config: Dict[str, Any] = Field(..., description="Model parameters")
    feature_config: Dict[str, Any] = Field(..., description="Feature engineering settings")
    data_config: Dict[str, Any] = Field(..., description="Data collection settings")
    training_config: Dict[str, Any] = Field(..., description="Training parameters")
    prediction_config: Dict[str, Any] = Field(..., description="Prediction settings")
    
    @field_validator('model_config')
    @classmethod
    def validate_model_config(cls, v):
        required_keys = ['default_model_type', 'ensemble_weights']
        if not all(key in v for key in required_keys):
            raise ValueError(f"Model config must contain {required_keys}")
        return v

# Utility functions for schema validation
def validate_dataframe_schema(df, required_columns: List[str]) -> bool:
    """Validate that DataFrame has required columns."""
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True

def create_feature_hash(feature_names: List[str]) -> str:
    """Create reproducible hash for feature list."""
    feature_str = '|'.join(sorted(feature_names))
    return hashlib.sha256(feature_str.encode()).hexdigest()

def validate_prediction_consistency(predictions: List[PredictionResult]) -> bool:
    """Validate that predictions are consistent (no duplicate ranks, etc.)."""
    ranks = [p.predicted_rank for p in predictions]
    if len(ranks) != len(set(ranks)):
        raise ValueError("Duplicate ranks found in predictions")
    
    if max(ranks) > len(predictions) or min(ranks) < 1:
        raise ValueError("Invalid rank range in predictions")
    
    return True 