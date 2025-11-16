"""
Model Evaluation Service for Comparing Baseline vs Upgraded Models

This module provides functionality to evaluate the impact of generated features
by comparing a baseline model (without generated features) against an upgraded
model (with generated features).

Workflow:
    1. Load and preprocess the original data
    2. Train baseline model on original features
    3. Apply generated features from a specific run
    4. Train upgraded model on original + generated features
    5. Compare performance metrics

Author: Aditya Deshmukh
"""

import pandas as pd
import xgboost as xgb
import pickle
from sqlalchemy.orm import Session
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
from typing import Dict, Any, Literal
from uuid import UUID

from app.services.preprocessor import Preprocessor
from app.services.genetic_programming import GeneticProgramming
from app.models import database_models


class ModelEvaluationService:
    """
    Service for evaluating the impact of generated features on model performance.
    
    This class compares a baseline XGBoost model (trained on original features only)
    against an upgraded XGBoost model (trained on original + generated features).
    
    Attributes:
        problem_type (str): Either "classification" or "regression"
        target_column (str): Name of the target variable
        baseline_model (xgb.XGBClassifier | xgb.XGBRegressor): Baseline model
        upgraded_model (xgb.XGBClassifier | xgb.XGBRegressor): Upgraded model
    """
    
    def __init__(self, problem_type: Literal["classification", "regression"], target_column: str):
        """
        Initialize the Model Evaluation Service.
        
        Args:
            problem_type: Either "classification" or "regression"
            target_column: Name of the target column in the dataset
        """
        self.problem_type = problem_type
        self.target_column = target_column
        self.baseline_model = None
        self.upgraded_model = None
    
    def run_comparison(
        self,
        dataframe: pd.DataFrame,
        dataset_id: UUID,
        run_id: UUID,
        db: Session
    ) -> Dict[str, Any]:
        """
        Run a complete comparison experiment between baseline and upgraded models.
        
        This method:
        1. Preprocesses the data
        2. Trains baseline model on original features
        3. Fetches generated features from the database
        4. Applies generated features to create upgraded dataset
        5. Trains upgraded model on combined features
        6. Compares and returns performance metrics
        
        Args:
            dataframe: Raw input DataFrame
            dataset_id: UUID of the dataset
            run_id: UUID of the feature generation run to use
            db: Database session for fetching generated features
            
        Returns:
            Dictionary containing comparison results:
            {
                "baseline_score": float,
                "upgraded_score": float,
                "improvement": float,
                "metric": str (e.g., "accuracy" or "rmse")
            }
        """
        # Step 1: Fetch the feature generation run to get parameters
        run = db.query(database_models.FeatureGenerationRun).filter(
            database_models.FeatureGenerationRun.id == run_id
        ).first()
        
        if not run:
            raise ValueError(f"Feature generation run with id {run_id} not found")
        
        if len(run.generated_features) == 0:
            raise ValueError(f"No generated features found for run {run_id}")
        
        # Step 2: Apply drop_columns if they were used during GP training
        drop_columns = run.parameters.get("drop_columns")
        if drop_columns:
            columns_to_drop = [col for col in drop_columns if col in dataframe.columns]
            dataframe = dataframe.drop(columns=columns_to_drop)
        
        # Step 3: Preprocess the original data
        preprocessor = Preprocessor(
            problem_type=self.problem_type,
            target_column=self.target_column
        )
        X_original, y = preprocessor.fit_transform(dataframe)
        
        # Step 4: Train and evaluate baseline model
        baseline_score = self._train_and_evaluate(X_original, y, "baseline")
        
        # Step 5: Generate new features from the stored GP programs
        # We'll use the best programs from the run to create new features
        print(f"📊 Generating {len(run.generated_features)} features from stored programs")
        
        # Load the fitted model to access the programs
        if not run.fitted_model:
            raise ValueError(f"No fitted model found for run {run_id}")
        
        fitted_gp_model = pickle.loads(run.fitted_model)
        
        # Get the best programs and apply them to create new features
        new_feature_columns = []
        
        # Get the programs from the fitted model
        if hasattr(fitted_gp_model, '_programs') and fitted_gp_model._programs:
            # Get the final generation programs
            final_generation = fitted_gp_model._programs[-1]
            # Take the top programs (up to 10)
            top_programs = final_generation[:min(10, len(final_generation))]
            
            for idx, program in enumerate(top_programs):
                if program is not None:
                    # Execute the program on the data
                    try:
                        feature_values = program.execute(X_original.values)
                        new_feature_columns.append(feature_values)
                    except Exception as e:
                        print(f"⚠️ Skipping feature {idx}: {e}")
                        continue
        
        if not new_feature_columns:
            raise ValueError(f"Could not generate any features from the stored programs")
        
        # Convert to DataFrame
        new_features_array = np.column_stack(new_feature_columns)
        new_features = pd.DataFrame(
            new_features_array,
            columns=[f"gp_feature_{i}" for i in range(len(new_feature_columns))],
            index=X_original.index
        )
        
        print(f"✅ Generated {new_features.shape[1]} new features")
        
        print(f"✅ Generated {new_features.shape[1]} new features")
        
        # Step 6: Combine original preprocessed features with new features
        # Align indices to ensure proper concatenation
        new_features_aligned = new_features.reindex(X_original.index)
        X_upgraded = pd.concat([X_original, new_features_aligned], axis=1)
        
        # Step 7: Train and evaluate upgraded model
        upgraded_score = self._train_and_evaluate(X_upgraded, y, "upgraded")
        
        # Step 8: Calculate improvement and return results
        improvement = upgraded_score - baseline_score
        
        metric_name = "accuracy" if self.problem_type == "classification" else "rmse"
        
        return {
            "baseline_score": float(baseline_score),
            "upgraded_score": float(upgraded_score),
            "improvement": float(improvement),
            "metric": metric_name,
            "num_original_features": X_original.shape[1],
            "num_generated_features": new_features_aligned.shape[1],
            "num_total_features": X_upgraded.shape[1]
        }
    
    def _train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str
    ) -> float:
        """
        Train an XGBoost model and evaluate its performance.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_type: Either "baseline" or "upgraded" (for tracking)
            
        Returns:
            Performance score (accuracy for classification, negative RMSE for regression)
        """
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y if self.problem_type == "classification" else None
        )
        
        # Create and train the appropriate XGBoost model
        if self.problem_type == "classification":
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)
            
            # Predict and calculate accuracy
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            
            print(f"✅ {model_type.capitalize()} Model - Accuracy: {score:.4f}")
            
        else:  # regression
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Predict and calculate RMSE
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # For regression, we return negative RMSE so that "higher is better"
            score = -rmse
            
            print(f"✅ {model_type.capitalize()} Model - RMSE: {rmse:.4f}")
        
        # Store the trained model
        if model_type == "baseline":
            self.baseline_model = model
        else:
            self.upgraded_model = model
        
        return score
