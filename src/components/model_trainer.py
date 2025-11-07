import os
import sys
from dataclasses import dataclass

# ML Models
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Custom Imports
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
        """
        Trains multiple ML models, compares them using R² score, and saves the best model.
        """
        try:
            logging.info("Splitting training and testing input data")

            # ✅ Split arrays into features (X) and target (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # ✅ Define models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # ✅ Define hyperparameter grids
            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                },
                "Random Forest": {
                    "n_estimators": [32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "n_estimators": [64, 128, 256],
                    "subsample": [0.7, 0.8, 0.9],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.05, 0.01],
                    "n_estimators": [64, 128, 256],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.05, 0.5],
                    "n_estimators": [32, 64, 128],
                },
            }

            # ✅ Evaluate all models
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            # ✅ Get best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name} (R² = {best_model_score:.3f})")

            if best_model_score < 0.6:
                raise CustomException("No suitable model found (R² < 0.6)")

            # ✅ Train best model fully
            best_model.fit(X_train, y_train)

            # ✅ Save trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # ✅ Evaluate best model on test data
            predictions = best_model.predict(X_test)
            r2_square = r2_score(y_test, predictions)

            print("\n🎯 Best Model:", best_model_name)
            print(f"✅ Test R² Score: {round(r2_square, 3)}")
            print(f"💾 Model saved at: {self.model_trainer_config.trained_model_file_path}\n")

            return r2_square

        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise CustomException(e, sys)
