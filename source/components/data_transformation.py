import pandas as pd
import numpy as np 
import os 
import sys
import dill
import logging  
from dataclasses import dataclass 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
#from source.exception import CustomError
from source.logger import logger
 
@dataclass



class DataTransformationConfig:
    preprocessor_obj_file_path = os.pth.join("artifacts", "preprocessor_obj.pkl")
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.preprocessor = None
        self.logger = logger

    def get_transformation_object(self):# we have done this pipeline to get the transformation object and transformation of numerical and categorical data
        try:
            numerical_columns = ["writing_score", "reading_score", "math_score"]
            #self.preprocessor = ColumnTransformer(
                #transformers=[
                                #("num", Pipeline(steps=[
                                    #("imputer", SimpleImputer(strategy="median")),
                                    #("scaler", StandardScaler())
                                #]), ["age", "fare"]),
                                #("cat", Pipeline(steps=[
                                    #("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                                    #("OneHot", OneHotEncoder(handle_unknown="ignore"))
                                #]), #["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"])
                            #]
                        #)
            categorical_features = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"] 
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('std_scaler', StandardScaler())
            ])
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('one_hot_encoding', OneHotEncoder(handle_unknown='ignore'))
            ])           
            logging.info("numerical columns standard scaling  computed") 
            logging.info("categorical columns one hot encoding computed")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('categorical_pipeline', categorical_pipeline, categorical_features)
                ])
            return preprocessor
        except Exception as e:
            
            raise e
   
    def initialize_and_save_preprocessor(self, trainpath,testpath):
        try:
            traindf=pd.read_csv(trainpath)
            testdf=pd.read_csv(testpath)
            logging.info("train and test data loaded")
            logging.info("obtaining preprocessor object")
            inputtrain_x=traindf.drop(columns=["math_score"])
            targettrain_y=traindf["math_score"]
            inputtest_x=testdf.drop(columns=["math_score"])
            targettest_y=testdf["math_score"]
            self.preprocessor = self.get_transformation_object()
            self.preprocessor.fit_transform(inputtrain_x)
            self.preprocessor.transform(inputtest_x)
            logging.info("preprocessor object has been obtained") 
            logging.info("preprocessor object has fit on train data")
            train_arr= np.c_[self.preprocessor.transform(inputtrain_x),targettrain_y]
            test_arr=np.c_[self.preprocessor.transform(inputtest_x),targettest_y]
            logging.info("saving and preprocessing has been done")
            return(train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)# wem  are  returning train test data also pickle file path
        except Exception as e:
            raise CustomException(e,sys)
            
