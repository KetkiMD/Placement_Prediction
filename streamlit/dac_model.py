import streamlit as st
import pandas as pd
import joblib
import io
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import boto3
import json
import os
            
# S3 bucket details
S3_BUCKET_NAME = "glueoutbucket"
S3_REGION = "us-east-1"
LAMBDA_FUNCTION_NAME = "modeloutput"  # Change this if your Lambda function has a different name
REGION_NAME = "us-east-1"

AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_SESSION_TOKEN = st.secrets.get("AWS_SESSION_TOKEN", None)  

cat_cols=['Grade',  'Apti_EC_Grade', 'Project_Grade', 'branch_cleaned']

theory_cols = ['OS_Theory','CPP_Theory','Java_Theory','DSA_Theory','DBT_Theory','WPT_Theory', 'WB_Java_Theory','DotNet_Theory','SDLC_Theory']
lab_cols = ['OS_Lab','CPP_Lab','Java_Lab','DSA_Lab','DBT_Lab','WPT_Lab','WB_Java_Lab','DotNet_Lab', 'SDLC_Lab']
percent_cols = ['10th_percentage', 'grad_percentage',  'Higher_Edu_Percent']
one_hot_col = ['branch_cleaned']
num_cols = ['10th_percentage',
 'grad_percentage',
 'OS_Theory',
 'OS_Lab',
 'CPP_Theory',
 'CPP_Lab',
 'Java_Theory',
 'Java_Lab',
 'DSA_Theory',
 'DSA_Lab',
 'DBT_Theory',
 'DBT_Lab',
 'WPT_Theory',
 'WPT_Lab',
 'WB_Java_Theory',
 'WB_Java_Lab',
 'DotNet_Theory',
 'DotNet_Lab',
 'SDLC_Theory',
 'SDLC_Lab',
 'Higher_Edu_Percent']




def preprocessing(X_df, preprocessor):
    X_transformed = preprocessor.fit_transform(X_df)
    all_col_names = num_cols + cat_cols


    X_df = pd.DataFrame(X_transformed, columns=all_col_names)

    degree_mapping = {
    "Computer": 5,
    "Electronics and Telecommunication": 4,
    "Electrical": 3,
    "Mechanical": 3,
    "Civil": 2,
    "BE": 2,
    "BSc": 2,
    "Mathematics": 2,
    "Chemical": 1,
    "Instrumentation": 1,
    "Physics": 1
}

# Apply encoding
    X_df["branch_cleaned"] = X_df["branch_cleaned"].map(degree_mapping)

    X_df[theory_cols] = X_df[theory_cols] / 40
    X_df[lab_cols] = X_df[lab_cols] / 60
    X_df[percent_cols] = X_df[percent_cols] / 100

    grade_cols = ['Grade', 'Apti_EC_Grade', 'Project_Grade']
    grade_mapping = {'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0, 'Fail': 0, 'Pass': 1}

    for col in grade_cols:
        if col in X_df.columns:
            X_df[col] = X_df[col].map(grade_mapping)

    return X_df



# Function to load a file from S3
def load_s3_file(folder, file_key):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        region_name=S3_REGION
    )
    obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=f"{folder}/{file_key}")
    return joblib.load(io.BytesIO(obj["Body"].read()))

# Function to load the model based on course
def load_model(course):
    if course == "DAC":
        return load_s3_file("models/dac", "latest_model.pkl")
    else:
        return None

# Function to load the preprocessor based on course
def load_preprocessor(course):
    if course == "DAC":
        return load_s3_file("models/dac", "latest_processor.pkl")
    else:
        return None

# Function to predict placement probability
def predict_student_placement(student_data, course):
    model = load_model(course)
    preprocessor = load_preprocessor(course)
    
    if model is None or preprocessor is None:
        return None
    
    student_df = pd.DataFrame([student_data])
    student_transformed = preprocessing(student_df,preprocessor)
    
    prediction = model.predict(student_transformed)
    probabilities = model.predict_proba(student_transformed)
    
    prob_yes = probabilities[0][list(model.classes_).index("Yes")]
    prob_no = probabilities[0][list(model.classes_).index("No")]

    return {
        "Placement Prediction": "Yes" if prediction[0] == "Yes" else "No",
        "Probability Yes": round(prob_yes * 100, 2),
        "Probability No": round(prob_no * 100, 2)
    }

# Function to calculate total marks
def calculate_total(theory, lab):
    return theory + lab






