"""
Genetic Programming Service for Automated Feature Engineering

This module leverages gplearn's SymbolicRegressor to evolve mathematical
expressions that serve as engineered features. It integrates seamlessly with
the Preprocessor for complete data preparation.

Workflow:
    1. Validate input data
    2. Preprocess using Preprocessor (datetime decomposition, encoding, scaling)
    3. Split into train/test sets
    4. Run genetic programming to evolve features
    5. Extract and return best programs

Author: Aditya Deshmukh
"""

from typing import Literal, List, Dict, Any, Tuple, Optional, Union
import pandas as pd
import pickle
from gplearn.genetic import SymbolicRegressor, SymbolicClassifier, SymbolicClassifier
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session
from uuid import UUID

from app.services.preprocessor import Preprocessor
from app.models import database_models


class GeneticProgramming:
    """
    Genetic Programming service for automated feature generation.
    
    This class uses gplearn's SymbolicRegressor to evolve mathematical
    expressions through genetic programming. The Preprocessor handles all
    data cleaning, transformation, and encoding automatically.
    
    Attributes:
        target_column (str): Name of the target variable
        problem_type (Literal["classification", "regression"]): Type of ML problem
        preprocessor (Preprocessor): Data preprocessing pipeline
        regressor (SymbolicRegressor): Fitted GP model (None until trained)
    """
    
    def __init__(
        self, 
        target_column: str, 
        problem_type: Literal["classification", "regression"],
        population_size: int = 1000,
        generations: int = 20,
        random_state: int = 42,
        drop_columns: List[str] | None = None
    ):
        """
        Initialize the Genetic Programming service.
        
        Args:
            target_column: Name of the target column in the dataset
            problem_type: Either "classification" or "regression"
            population_size: Number of programs in each generation (default: 1000)
            generations: Number of generations to evolve (default: 20)
            random_state: Random seed for reproducibility (default: 42)
            
        Raises:
            ValueError: If problem_type is not "classification" or "regression"
        """
        if problem_type not in {"classification", "regression"}:
            raise ValueError("problem_type must be 'classification' or 'regression'")
            
        self.target_column = target_column
        self.problem_type = problem_type
        self.population_size = population_size
        self.generations = generations
        self.random_state = random_state
        self.drop_columns = drop_columns
        
        # Initialize preprocessor
        self.preprocessor = Preprocessor(
            problem_type=problem_type, 
            target_column=target_column
        )
        
        # Will be set after training
        self.regressor = None
        self._feature_names = None
    
    def run(
        self, 
        dataframe: pd.DataFrame,
        db: Optional[Session] = None,
        dataset_id: Optional[UUID] = None
    ) -> Union[List[Dict[str, Any]], database_models.FeatureGenerationRun]:
        """
        Execute the complete genetic programming workflow.
        
        This method:
        1. Validates the input dataframe
        2. Preprocesses data using the Preprocessor
        3. Splits data into train/test sets
        4. Trains the SymbolicRegressor
        5. Extracts evolved features
        6. Saves results to database (if db session provided)
        
        Args:
            dataframe: Input DataFrame containing features and target
            db: Database session for saving results (optional)
            dataset_id: ID of the dataset being processed (optional, required if db provided)
            
        Returns:
            If db provided: FeatureGenerationRun object with nested generated_features
            If db not provided: List of dictionaries containing evolved feature information
            
        Raises:
            ValueError: If target column not found or preprocessing fails
        """
        # Drop columns if specified
        if self.drop_columns:
            # Ensure columns exist before dropping
            columns_to_drop = [col for col in self.drop_columns if col in dataframe.columns]
            dataframe = dataframe.drop(columns=columns_to_drop)

        # Validate dataframe before preprocessing
        if self.target_column not in dataframe.columns:
            raise ValueError(f"Target column '{self.target_column}' not in dataframe. Available columns: {list(dataframe.columns)}")
        
        # Step 1: Validate input
        self._validate_input(dataframe)
        
        # Step 2: Preprocess data
        # Preprocessor handles:
        # - Datetime decomposition
        # - Missing value imputation
        # - Scaling (numeric features)
        # - One-hot encoding (categorical features)
        # - Target encoding (classification) or conversion (regression)
        X_processed, y_processed = self.preprocessor.fit_transform(dataframe)
        
        # Store feature names for later reference
        self._feature_names = X_processed.columns.tolist()
        
        # Step 3: Validate processed data
        self._validate_processed_data(X_processed, y_processed)
        
        # Step 4: Split data
        X_train, X_test, y_train, y_test = self._split_data(
            X_processed, 
            y_processed
        )
        
        # Step 5: Train genetic programming model
        self._train_gp_model(X_train, y_train)
        
        # Step 6: Extract and return best features
        best_features = self._extract_best_features()
        
        # Step 7: Save to database if session provided
        if db is not None and dataset_id is not None:
            return self._save_to_warehouse(db, dataset_id, best_features)
        
        return best_features

    def _validate_input(self, dataframe: pd.DataFrame) -> None:
        """
        Validate the input dataframe.
        
        Args:
            dataframe: Input DataFrame to validate
            
        Raises:
            ValueError: If dataframe is empty or invalid
        """
        if dataframe.empty:
            raise ValueError("Input dataframe is empty")
        
        if len(dataframe) < 10:
            raise ValueError(f"Dataset too small. Need at least 10 rows, got {len(dataframe)}")
    
    def _validate_processed_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Validate the processed data before training.
        
        Args:
            X: Processed feature DataFrame
            y: Processed target Series
            
        Raises:
            ValueError: If data is invalid for training
        """
        if X.empty:
            raise ValueError("No features available after preprocessing")
        
        if len(X) != len(y):
            raise ValueError(f"Feature and target lengths don't match: {len(X)} vs {len(y)}")
        
        # Check for NaN or inf values
        if X.isnull().any().any():
            raise ValueError("Processed features contain NaN values")
        
        if y.isnull().any():
            raise ValueError("Processed target contains NaN values")
    
    def _split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y if self.problem_type == "classification" else None
        )
        
        return X_train, X_test, y_train, y_test
    
    def _train_gp_model(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> None:
        """
        Initialize and train the correct GP model based on problem_type.
        """

        # --- THIS IS THE SMART SWITCH ---
        if self.problem_type == "classification":
            print("--- Initializing SymbolicClassifier ---") # For debugging
            self.regressor = SymbolicClassifier(
                population_size=self.population_size,
                generations=self.generations,
                stopping_criteria=0.01,
                p_crossover=0.7,
                p_subtree_mutation=0.1,
                p_hoist_mutation=0.05,
                p_point_mutation=0.1,
                max_samples=0.9,
                verbose=1,
                parsimony_coefficient=0.01,
                random_state=self.random_state,
                n_jobs=-1,
                feature_names=self._feature_names
            )
        else: # This is the "regression" case
            print("--- Initializing SymbolicRegressor ---") # For debugging
            self.regressor = SymbolicRegressor(
                population_size=self.population_size,
                generations=self.generations,
                stopping_criteria=0.01,
                p_crossover=0.7,
                p_subtree_mutation=0.1,
                p_hoist_mutation=0.05,
                p_point_mutation=0.1,
                max_samples=0.9,
                verbose=1,
                parsimony_coefficient=0.01,
                random_state=self.random_state,
                n_jobs=-1,
                feature_names=self._feature_names
            )

        # Fit the model
        self.regressor.fit(X_train.values, y_train.values)

    def _extract_best_features(self) -> List[Dict[str, Any]]:
        """
        Extract the best features from the trained GP model.
        
        Returns:
            List of feature dictionaries containing:
            - feature_id: Unique identifier
            - expression: Mathematical expression as string
            - fitness: Fitness score (lower is better)
            - description: Human-readable description
        """
        best_features = []
        
        # Extract top programs from the final generation
        if hasattr(self.regressor, '_programs') and self.regressor._programs:
            final_generation = self.regressor._programs[-1]
            
            # Get top 10 programs (or fewer if not available)
            top_programs = final_generation[:min(10, len(final_generation))]
            
            for idx, program in enumerate(top_programs):
                if program is not None:
                    feature_info = self._create_feature_dict(
                        feature_id=idx + 1,
                        program=program,
                        description="Evolved feature from final generation"
                    )
                    best_features.append(feature_info)
        
        # Fallback: use the best overall program if no programs in final generation
        if not best_features and hasattr(self.regressor, '_program'):
            program = self.regressor._program
            if program is not None:
                feature_info = self._create_feature_dict(
                    feature_id=1,
                    program=program,
                    description="Best overall evolved feature"
                )
                best_features.append(feature_info)
        
        return best_features
    
    def _create_feature_dict(
        self, 
        feature_id: int, 
        program: Any,
        description: str
    ) -> Dict[str, Any]:
        """
        Create a feature dictionary from a program.
        
        Args:
            feature_id: Unique identifier for the feature
            program: gplearn program object
            description: Description of the feature
            
        Returns:
            Dictionary containing feature information
        """
        expression = str(program)
        fitness = getattr(program, 'fitness_', None)
        
        return {
            "feature_id": feature_id,
            "expression": expression,
            "fitness": float(fitness) if fitness is not None else None,
            "description": description,
            "feature_names": self._feature_names
        }
    
    def get_preprocessor(self) -> Preprocessor:
        """
        Get the fitted preprocessor for reuse.
        
        Returns:
            Fitted Preprocessor instance
        """
        return self.preprocessor
    
    def predict_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Apply evolved features to new data.
        
        Args:
            dataframe: New DataFrame to transform
            
        Returns:
            DataFrame with evolved features applied
            
        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if self.regressor is None:
            raise RuntimeError("Model must be trained before predicting. Call run() first.")
        
        # Use preprocessor's transform method (not fit_transform)
        X_processed, _ = self.preprocessor.transform(dataframe)
        
        # Apply evolved features
        evolved_features = self.regressor.transform(X_processed.values)
        
        return pd.DataFrame(
            evolved_features,
            columns=[f"gp_feature_{i}" for i in range(evolved_features.shape[1])],
            index=dataframe.index
        )
    
    def _save_to_warehouse(
        self,
        db: Session,
        dataset_id: UUID,
        best_features: List[Dict[str, Any]]
    ) -> database_models.FeatureGenerationRun:
        """
        Save the feature generation run and features to the database warehouse.
        
        Args:
            db: Database session
            dataset_id: ID of the dataset used for feature generation
            best_features: List of generated features to save
            
        Returns:
            The created FeatureGenerationRun object with nested generated_features
        """
        # Pickle the fitted model for later use
        if self.regressor is None:
            raise ValueError("Cannot save warehouse: regressor is None. Model must be trained before saving.")
        
        pickled_model = pickle.dumps(self.regressor)
        print(f"📦 Pickled model size: {len(pickled_model)} bytes")
        
        # Create FeatureGenerationRun record
        run = database_models.FeatureGenerationRun(
            dataset_id=dataset_id,
            parameters={
                "target_column": self.target_column,
                "problem_type": self.problem_type,
                "population_size": self.population_size,
                "generations": self.generations,
                "random_state": self.random_state,
                "drop_columns": self.drop_columns
            },
            fitted_model=pickled_model  # Save the pickled model
        )
        db.add(run)
        db.flush()  # Flush to get the run.id
        
        print(f"🆕 Created NEW run with ID: {run.id}")
        
        # Create GeneratedFeature records for each feature
        for feature in best_features:
            generated_feature = database_models.GeneratedFeature(
                run_id=run.id,
                expression=feature["expression"],
                fitness=feature["fitness"],
                feature_names=feature["feature_names"]
            )
            db.add(generated_feature)
        
        # Commit all changes
        db.commit()
        db.refresh(run)  # Refresh to load the generated_features relationship
        print(f"✅ Saved {len(best_features)} features to warehouse (run_id: {run.id})")
        
        return run