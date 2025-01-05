import os
import sys
import pickle
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV  

def save_object(file_path, obj):
    try:
        # Check if the directory exists, if not, create it
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        # Raise a custom exception if an error occurs
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        # Loop through each model
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            # Perform GridSearchCV for hyperparameter tuning
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Set the best params and train the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict and calculate R2 score for training and test set
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the model's score in the report
            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        # Load the saved object (model or preprocessor)
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
