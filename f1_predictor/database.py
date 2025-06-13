# f1_predictor/database.py - Database layer using DuckDB

import os
import logging
import duckdb
import json
import pickle
from typing import Optional, Any, List, Tuple
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class F1Database:
    """Simple database layer for F1 prediction system using DuckDB."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection."""
        self.db_path = db_path or 'f1_data_real/f1_prediction.db'
        self._ensure_db_directory()
        self.conn = duckdb.connect(self.db_path)
        self._create_tables()
    
    def _ensure_db_directory(self):
        """Ensure database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)
    
    def _create_tables(self):
        """Create basic database tables."""
        try:
            # Sessions table for raw data
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY,
                    year INTEGER,
                    round_number INTEGER,
                    circuit VARCHAR,
                    session_type VARCHAR,
                    data JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Models table for storing model metadata
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY,
                    model_type VARCHAR,
                    feature_hash VARCHAR,
                    performance JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
    
    def store_session_data(self, year: int, round_num: int, circuit: str, session_type: str, data: dict) -> bool:
        """Store session data."""
        try:
            self.conn.execute("""
                INSERT INTO sessions (year, round_number, circuit, session_type, data)
                VALUES (?, ?, ?, ?, ?)
            """, [year, round_num, circuit, session_type, json.dumps(data)])
            return True
        except Exception as e:
            logger.error(f"Failed to store session data: {e}")
            return False
    
    def store_feature_matrix(self, feature_matrix: Any, data_df: pd.DataFrame) -> bool:
        """Store feature matrix and data."""
        try:
            # Save DataFrame to parquet file
            data_dir = os.path.join(os.path.dirname(self.db_path), 'feature_data')
            os.makedirs(data_dir, exist_ok=True)
            
            data_filename = f"features_{feature_matrix.hash}.parquet"
            data_path = os.path.join(data_dir, data_filename)
            data_df.to_parquet(data_path)
            
            # Store metadata in database
            self.conn.execute("""
                INSERT OR REPLACE INTO feature_matrices 
                (feature_hash, feature_names, target_column, shape, data_path)
                VALUES (?, ?, ?, ?, ?)
            """, [
                feature_matrix.hash,
                json.dumps(feature_matrix.feature_names),
                feature_matrix.target_column,
                json.dumps(feature_matrix.shape),
                data_path
            ])
            
            logger.info(f"Stored feature matrix: {feature_matrix.hash}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store feature matrix: {e}")
            return False
    
    def load_feature_matrix(self, feature_hash: str) -> Optional[Tuple[Any, pd.DataFrame]]:
        """Load feature matrix by hash."""
        try:
            result = self.conn.execute("""
                SELECT feature_hash, feature_names, target_column, shape, data_path
                FROM feature_matrices 
                WHERE feature_hash = ?
            """, [feature_hash]).fetchone()
            
            if not result:
                return None
            
            feature_hash, feature_names_json, target_column, shape_json, data_path = result
            
            # Load DataFrame from parquet
            if not os.path.exists(data_path):
                logger.warning(f"Feature data file not found: {data_path}")
                return None
            
            data_df = pd.read_parquet(data_path)
            
            # Reconstruct FeatureMatrix
            feature_matrix = FeatureMatrix(
                features=data_df.to_dict('records'),
                feature_names=json.loads(feature_names_json),
                target_column=target_column,
                shape=tuple(json.loads(shape_json)),
                hash=feature_hash
            )
            
            return feature_matrix, data_df
            
        except Exception as e:
            logger.error(f"Failed to load feature matrix: {e}")
            return None
    
    def store_model(self, model_metadata: Any, model_obj: Any) -> bool:
        """Store trained model with metadata."""
        try:
            # Save model object to pickle file
            models_dir = os.path.join(os.path.dirname(self.db_path), 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            model_filename = f"model_{model_metadata.model_type}_{model_metadata.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            model_path = os.path.join(models_dir, model_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_obj, f)
            
            # Store metadata in database
            self.conn.execute("""
                INSERT INTO models 
                (model_type, algorithm, feature_hash, feature_names, training_date, 
                 hyperparameters, performance_metrics, imputation_values, model_path, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                model_metadata.model_type,
                model_metadata.algorithm,
                model_metadata.feature_hash,
                json.dumps(model_metadata.feature_names),
                model_metadata.training_date,
                json.dumps(model_metadata.hyperparameters),
                json.dumps(model_metadata.performance_metrics),
                json.dumps(model_metadata.imputation_values),
                model_path,
                model_metadata.version
            ])
            
            model_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            logger.info(f"Stored model: {model_id} ({model_metadata.model_type})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store model: {e}")
            return False
    
    def load_latest_model(self, model_type: str) -> Optional[Tuple[Any, Any]]:
        """Load latest active model of specified type."""
        try:
            result = self.conn.execute("""
                SELECT id, model_type, algorithm, feature_hash, feature_names, training_date,
                       hyperparameters, performance_metrics, imputation_values, model_path, version
                FROM models 
                WHERE model_type = ? AND is_active = TRUE
                ORDER BY training_date DESC
                LIMIT 1
            """, [model_type]).fetchone()
            
            if not result:
                return None
            
            (model_id, model_type, algorithm, feature_hash, feature_names_json, training_date,
             hyperparams_json, metrics_json, imputation_json, model_path, version) = result
            
            # Load model object
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                return None
            
            with open(model_path, 'rb') as f:
                model_obj = pickle.load(f)
            
            # Reconstruct metadata
            class SimpleMetadata:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
                        
            metadata = SimpleMetadata(
                model_type=model_type,
                algorithm=algorithm,
                feature_hash=feature_hash,
                feature_names=json.loads(feature_names_json),
                training_date=training_date,
                hyperparameters=json.loads(hyperparams_json),
                performance_metrics=json.loads(metrics_json),
                imputation_values=json.loads(imputation_json),
                version=version
            )
            
            return metadata, model_obj
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def store_predictions(self, race_weekend_id: str, predictions: List[Any], model_id: Optional[int] = None) -> bool:
        """Store prediction results."""
        try:
            for pred in predictions:
                self.conn.execute("""
                    INSERT INTO predictions 
                    (race_weekend_id, driver, team, predicted_position, predicted_rank, 
                     confidence_score, model_type, model_id, prediction_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    race_weekend_id,
                    pred.driver,
                    pred.team,
                    pred.predicted_position,
                    pred.predicted_rank,
                    pred.confidence_score,
                    pred.model_type,
                    model_id,
                    pred.prediction_timestamp
                ])
            
            logger.info(f"Stored {len(predictions)} predictions for weekend {race_weekend_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store predictions: {e}")
            return False
    
    def get_session_data(self, year: int, round_number: int, session_type: str) -> Optional[Any]:
        """Get session data by year, round, and type."""
        try:
            result = self.conn.execute("""
                SELECT session_type, year, round_number, circuit, session_date, weather_data, results
                FROM raw_sessions 
                WHERE year = ? AND round_number = ? AND session_type = ?
            """, [year, round_number, session_type]).fetchone()
            
            if not result:
                return None
            
            session_type, year, round_number, circuit, session_date, weather_json, results_json = result
            
            class SimpleSessionData:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
                        
            return SimpleSessionData(
                session_type=session_type,
                year=year,
                round_number=round_number,
                circuit=circuit,
                session_date=session_date,
                weather_data=json.loads(weather_json) if weather_json else None,
                results=json.loads(results_json)
            )
            
        except Exception as e:
            logger.error(f"Failed to get session data: {e}")
            return None
    
    def get_historical_data(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Get historical data as DataFrame."""
        try:
            query = """
                SELECT year, round_number, circuit, session_type, data
                FROM sessions 
                WHERE year BETWEEN ? AND ?
                ORDER BY year, round_number
            """
            
            result = self.conn.execute(query, [start_year, end_year]).fetchall()
            
            # Convert to DataFrame
            data_rows = []
            for year, round_num, circuit, session_type, data_json in result:
                data = json.loads(data_json)
                row = {
                    'Year': year,
                    'Race_Num': round_num,
                    'Circuit': circuit,
                    'Session_Type': session_type,
                    **data
                }
                data_rows.append(row)
            
            return pd.DataFrame(data_rows)
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data to maintain database size."""
        try:
            cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
            
            # Clean old predictions
            self.conn.execute("""
                DELETE FROM predictions 
                WHERE created_at < ?
            """, [cutoff_date])
            
            # Clean old feature matrices (keep only recent ones)
            self.conn.execute("""
                DELETE FROM feature_matrices 
                WHERE created_at < ?
            """, [cutoff_date])
            
            # Mark old models as inactive instead of deleting
            self.conn.execute("""
                UPDATE models 
                SET is_active = FALSE 
                WHERE created_at < ? AND is_active = TRUE
            """, [cutoff_date])
            
            logger.info(f"Cleaned up data older than {days_to_keep} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

# Global instance
_db_instance = None

def get_database() -> F1Database:
    """Get global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = F1Database()
    return _db_instance 