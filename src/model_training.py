from src.model_params import RANDOM_SEARCH_PARAMS
from numpy import average
import os 
import joblib
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from src.model_params import *


import mlflow
# import mlflow.sklearn

from src.logger import get_logger
from src.custom_exception import CustomException 

logger = get_logger(__name__)

class ModelTraining():
    def __init__(self, processed_data_path, model_output_path):
        self.processed_path = processed_data_path
        self.model_path = model_output_path
        self.clf = None
        self.params_dict = LR_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

        os.makedirs(self.model_path, exist_ok=True)
        logger.info("Model Training Initialized...")

    def load_data(self):
        try:
            self.X_train = joblib.load(os.path.join(self.processed_path, "X_train.pkl"))
            self.X_test = joblib.load(os.path.join(self.processed_path, "X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_path, "y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_path, "y_test"))
            
            logger.info("Data loaded successfully.")

        except Exception as e:
            logger.error(f"Error while loading data: {e}")
            raise CustomException("Failed to load data", e)

        
    def train_model(self):
        try:
            logger.info("Initializing our model")

            lr = LogisticRegression(**self.params_dict)
            self.clf.fit(self.X_train, self.y_train)

            joblib.dump(self.clf, os.path.join(self.model_path, "model.pkl"))

            logger.info("Model trained and saved successfully.")

        except Exception as e: 
            logger.error(f"Error while training model {e}")
            raise CustomException("Failed to train model", e)

        
    def evaluate_model(self):
        try:
            y_pred = self.clf.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test. y_pred, average="weighted")
            recall = recall_score(self.y_test, y_pred, average="weighted")
            f1 = f1_score(self.y_test, y_pred, average="weighted")

            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"F1 Score: {f1}")

            logger.info("Model Evaluation Done...")

        except Exception as e:
            logger.error(f"Error while evaluating model: {e}")
            raise CustomException("Failed to evaluate model", e)

    
    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()


if __name__ == "__main__":
    trainer = ModelTraining("artifacts/processed/", "artifacts/models/")
    trainer.run()
