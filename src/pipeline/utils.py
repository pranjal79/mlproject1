import os
import sys
import dill
from sklearn.metrics import r2_score
from src.pipeline.exception import CustomException
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    """
    Saves a Python object to the specified file path using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param=None):
    """
    Trains and evaluates multiple ML models using GridSearchCV (if params given).
    Returns a dictionary with model names and their R² scores.
    """
    try:
        report = {}

        for model_name, model in models.items():
            print(f"🔍 Training {model_name}...")

            # Hyperparameter tuning if parameters provided
            if param and model_name in param and len(param[model_name]) > 0:
                gs = GridSearchCV(model, param[model_name], cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)

            # Train model
            model.fit(X_train, y_train)

            # Predict
            y_pred_test = model.predict(X_test)

            # Calculate R²
            test_score = r2_score(y_test, y_pred_test)
            report[model_name] = test_score

            print(f"✅ {model_name} R² Score: {round(test_score, 3)}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
