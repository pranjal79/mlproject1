import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.pipeline.exception import CustomException
from src.pipeline.logger import logging

# ✅ Import from correct modules
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        print("➡️ Step 1: Starting data ingestion...")

        try:
            # ✅ Read dataset
            df = pd.read_csv('notebook/data/student.csv')
            print("✅ Step 2: Loaded student.csv successfully.")
            logging.info("Read the dataset as dataframe")

            # ✅ Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            print("📁 Step 3: Created 'artifacts' folder (if not already present).")

            # ✅ Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            print(f"💾 Step 4: Saved raw data to {self.ingestion_config.raw_data_path}")

            # ✅ Train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            print("✂️ Step 5: Split data into train and test sets.")

            # ✅ Save split data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            print(f"✅ Step 6: Train file saved at {self.ingestion_config.train_data_path}")
            print(f"✅ Step 7: Test file saved at {self.ingestion_config.test_data_path}")

            logging.info("Ingestion of the data is completed")
            print("🎉 Step 8: Data ingestion completed successfully!")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            print("❌ ERROR: Something went wrong during ingestion!")
            print("Details:", e)
            logging.error(f"Error occurred in data ingestion: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # ✅ Step 1: Data Ingestion
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()

        # ✅ Step 2: Data Transformation
        from src.components.data_transformation import DataTransformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)
        print("🔧 Data Transformation Completed!")
        print("📦 Preprocessor saved at:", preprocessor_path)

        # ✅ Step 3: Model Training
        from src.components.model_trainer import ModelTrainer
        model_trainer = ModelTrainer()
        r2_square = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(f"🎯 Model training completed successfully! Final R² Score: {r2_square:.4f}")

    except Exception as e:
        print("❌ Pipeline Execution Failed!")
        print("Details:", e)

