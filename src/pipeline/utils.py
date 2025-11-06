import os
import sys
import dill
from src.pipeline.exception import CustomException


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


if __name__ == "__main__":
    print("✅ utils.py loaded successfully with save_object() defined!")
