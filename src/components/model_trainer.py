import os
import sys
from dataclasses import dataclass

# ML models
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Custom modules
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.pipeline.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")

            # Split input and target features
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Hyperparameter grids
            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "subsample": [0.7, 0.8, 0.9],
                    "n_estimators": [32, 64, 128],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "n_estimators": [32, 64, 128],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "n_estimators": [32, 64, 128],
                },
            }

            # Evaluate all models (GridSearchCV inside utils.py)
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            # Pick the best model based on R² score
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name} (R² = {best_model_score:.3f})")

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R² > 0.6")

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # Evaluate on test set
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            print("\n🎯 Best Model:", best_model_name)
            print("✅ Test R² Score:", round(r2_square, 3))
            print(f"💾 Model saved at: {self.model_trainer_config.trained_model_file_path}")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
