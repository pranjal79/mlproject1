import os
import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.pipeline.exception import CustomException


def save_object(file_path, obj):
    """
    Saves any Python object (like model or preprocessor) using dill.
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
    Trains and evaluates multiple ML models using GridSearchCV for hyperparameter tuning.
    
    Args:
        X_train (ndarray): Training features
        y_train (ndarray): Training target
        X_test (ndarray): Testing features
        y_test (ndarray): Testing target
        models (dict): Dictionary of model_name: model_instance
        param (dict): Dictionary of model_name: parameter_grid
    
    Returns:
        dict: Model name -> R² score on test set
    """
    try:
        report = {}

        for model_name, model in models.items():
            print(f"\n🔍 Training and tuning {model_name}...")

            # Get parameter grid for current model
            param_grid = param.get(model_name, {}) if param else {}

            # Apply GridSearchCV if parameters are defined
            if param_grid:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    n_jobs=-1,
                    verbose=0
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                print(f"   🔧 Best Params for {model_name}: {gs.best_params_}")
            else:
                # No params — fit directly
                model.fit(X_train, y_train)
                best_model = model

            # Predict on test data
            y_pred_test = best_model.predict(X_test)

            # Evaluate R²
            test_score = r2_score(y_test, y_pred_test)
            report[model_name] = test_score

            print(f"✅ {model_name} | R² Score: {round(test_score, 3)}")

        print("\n🏁 Model evaluation completed!\n")
        return report

    except Exception as e:
        raise CustomException(e, sys)
