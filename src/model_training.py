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
import mlflow.sklearn

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


    def load_data(self):
        try:
            self.X_train = joblib.load(os.path.join(self.processed_path, "X_train.pkl"))
            self.X_test = joblib.load(os.path.join(self.processed_path, "X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_path, "y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_path, "y_test.pkl"))
            
            logger.info("Data loaded successfully.")

            return self.X_train, self.y_train, self.X_test, self.y_test

        except Exception as e:
            logger.error(f"Error while loading data: {e}")
            raise CustomException("Failed to load data", e)

        
    def train_model(self):
        try:
            logger.info("Initializing our model")

            self.clf = LogisticRegression(multi_class='ovr')
            
            logger.info("Starting our Hyperparameter tuning")

            random_search = RandomizedSearchCV(
                estimator = self.clf,
                param_distributions=self.params_dict,
                n_iter = self.random_search_params["n_iter"],
                scoring = self.random_search_params["scoring"],
                cv = self.random_search_params["cv"],
                verbose = self.random_search_params["verbose"],
                n_jobs = self.random_search_params["n_jobs"],
                random_state = self.random_search_params["random_state"],
            )
            
            random_search.fit(self.X_train, self.y_train)

            logger.info("Hyperparameter Tuning Completed")

            best_params = random_search.best_params_
            best_lr = random_search.best_estimator_  

            logger.info(f"Best parameters are: {best_params}")

            return best_lr

        except Exception as e: 
            logger.error(f"Error while training model {e}")
            raise CustomException("Failed to train model", e)

        
    def evaluate_model(self):
        try:
            logger.info("Evaluating our model")

            y_pred = self.clf.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average="weighted")
            recall = recall_score(self.y_test, y_pred, average="weighted")
            f1 = f1_score(self.y_test, y_pred, average="weighted")

            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"F1 Score: {f1}")

            logger.info("Model Evaluation Done...")

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

        except Exception as e:
            logger.error(f"Error while evaluating model: {e}")
            raise CustomException("Failed to evaluate model", e)


    def save_model(self, model):
        try:
            joblib.dump(model, os.path.join(self.model_path, "model.pkl"))
            logger.info("Model saved successfully.")

        except Exception as e:
            logger.error(f"Error while saving model: {e}")
            raise CustomException("Failed to save model", e)


    def run(self):
        try:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("SM_Machine_Efficiency_Prediction")
            with mlflow.start_run():
                logger.info("Starting our Model Training")

                logger.info("Starting our MLFLOW Experimentation")
                logger.info("Logging the training and testing dataset to MLFLOW")
                mlflow.log_artifact(os.path.join(self.processed_path, "X_train.pkl"), artifact_path="datasets")
                mlflow.log_artifact(os.path.join(self.processed_path, "X_test.pkl"), artifact_path="datasets")
                mlflow.log_artifact(os.path.join(self.processed_path, "y_train.pkl"), artifact_path="datasets")
                mlflow.log_artifact(os.path.join(self.processed_path, "y_test.pkl"), artifact_path="datasets")

                X_train, y_train, X_test, y_test = self.load_data()
                best_lr = self.train_model()
                self.clf = best_lr
                metrics = self.evaluate_model()
                self.save_model(best_lr)

                logger.info("Logging the model into MLFLOW")

                # The recommended way is to use mlflow.sklearn.log_model for scikit-learn models
                mlflow.sklearn.log_model(best_lr, "model")
                # Also logging the raw pkl file as requested by the initial structure
                mlflow.log_artifact(os.path.join(self.model_path, "model.pkl"), artifact_path="raw_models")
                
                logger.info("Logging Params and metrics to MLFLOW")
                mlflow.log_params(best_lr.get_params())
                mlflow.log_metrics(metrics)

                logger.info("Model Training Successfully Completed")

        except Exception as e:
            logger.error(f"Error in  model training pipeline: {e}")
            raise CustomException("Failed during model training pipeline", e)


if __name__ == "__main__":
    trainer = ModelTraining("artifacts/processed/", "artifacts/models/")
    trainer.run()
