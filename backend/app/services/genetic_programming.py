import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
import numpy as np

class GeneticProgrammingService:
    def __init__(self, target_column: str):
        self.target_column = target_column
        self.regressor = None
    
    def run(self, dataframe: pd.DataFrame) -> list:
        try:
            if self.target_column not in dataframe.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in dataset")
            
            X = dataframe.drop(columns=[self.target_column])
            y = dataframe[self.target_column]
            
            if y.dtype == 'object' or y.dtype.name == 'category':       
                mapping = {'Y': 1, 'N': 0}
                # Apply the translation.
                y = y.map(mapping)
            
# Handle non-numeric columns
            
            numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_columns) == 0:
                raise ValueError("No numeric columns found for feature generation")
            
            X = X[numeric_columns]
            
            X = X.fillna(0)
            y = y.fillna(0)

            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            self.regressor = SymbolicRegressor(
                population_size=1000,
                generations=20,
                stopping_criteria=0.01,
                p_crossover=0.7,
                p_subtree_mutation=0.1,
                p_hoist_mutation=0.05,
                p_point_mutation=0.1,
                max_samples=0.9,
                verbose=1,
                parsimony_coefficient=0.01,
                random_state=42,
                n_jobs=-1,
                feature_names=X.columns.tolist()
            )
            
            self.regressor.fit(X_train.values, y_train.values)
            
            best_features = []
            
            if hasattr(self.regressor, '_programs'):
                for i, program in enumerate(self.regressor._programs[-1][:10]):  # Top 10
                    if program:
                        feature_expression = str(program)
                        fitness = program.fitness_
                        best_features.append({
                            "feature_id": i + 1,
                            "expression": feature_expression,
                            "fitness": float(fitness) if fitness else None,
                            "description": f"Generated feature using genetic programming"
                        })
            
            # If no programs available, return the best overall program
            if not best_features and hasattr(self.regressor, '_program'):
                best_features.append({
                    "feature_id": 1,
                    "expression": str(self.regressor._program),
                    "fitness": float(self.regressor._program.fitness_) if hasattr(self.regressor._program, 'fitness_') else None,
                    "description": "Best generated feature"
                })
            
            return best_features
            
        except Exception as e:
            raise ValueError(f"Error in genetic programming: {str(e)}")