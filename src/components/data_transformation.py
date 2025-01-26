import pandas as pd
import numpy as np 
import os 
import sys
import logging  
 from dataclasses import dataclass 
 from sklearn.compose import ColumnTransformer
 import sklearn.impute import SimpleImputer
 from sklearn.pipeline import Pipeline 
 from sklearn.preprocessing import StandardScaler, OneHotEncoder 
 from src.exceptions import CustomError
 from src.logger import logger
 
@dataclass



class DataTransformationConfig:
    preprocessor_obj_file_path = os.pth.join("artifacts", "preprocessor_obj.pkl")
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.preprocessor = None
        self.logger = logger

    def fit(self, data):
        try:
            self.logger.info("Fitting the data transformation pipeline")
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("num", Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())
                    ]), ["age", "fare"]],
                    ("cat", Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    def get_data_transformation_object(self):
        try:
            numrical_columns =["writing_score", "reading_score", "math_score"]
            categorical_columns = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
        except Exception as e:
            raise e                    