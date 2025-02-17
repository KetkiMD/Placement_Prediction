import pandas as pd
import boto3
import warnings
from io import StringIO
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')



bucket_name = "glueoutbucket"
folder_path = "data/dbda/previous/"

s3 = boto3.client("s3")

s3.put_object(Bucket=bucket_name, Key=folder_path)


path = 's3://outputdac/dbda/part-00000-8c586924-f936-4a7a-8094-43eb0eb19789-c000.csv'


def data_ingestion(path):
    drop_columns = ['form_number', 'Percentage','PRN', 'dob', 'course', 'grad_degree', 'pg_percentage', 'pre_ccat', 'Company_Name', 'Package', 'month', 'year','DBMSTotal',
 'JavaTotal', 'PythonRTotal','branch', 'StatsTotal', 'DataVizTotal', 'BigDataTotal', 'LinuxCloudTotal', 'MLTotal', 'Total',
'Company_Name', 'Package', 'year', 'month','Age','DBMSStatus', 'JavaStatus', 'PythonRStatus', 'StatsStatus', 'DataVizStatus',
 'BigDataStatus', 'LinuxCloudStatus', 'MLStatus','CCATrank','Result']
    df = pd.read_csv(path)
    df = df.drop(drop_columns,axis=1)
    X_df = df.drop(['Is_Placed'],axis=1)
    y = df['Is_Placed']
    return X_df,y



X_df,y = data_ingestion(path)


num_cols = ['10th_percentage', 'grad_percentage', 'DBMSTheory', 'DBMSLab', 'JavaTheory',
                'JavaLab', 'PythonRTheory', 'PythonRLab', 'StatsTheory', 'StatsLab', 'DataVizTheory', 'DataVizLab',
                'BigDataTheory', 'BigDataLab', 'LinuxCloudTheory', 'LinuxCloudLab', 'MLTheory', 'MLLab'
                , 'Higher_Edu_Percent']
    
cat_cols = ['Grade', 'aptigrade', 'projectgrade', 'branch_cleaned']
    
theory_cols = ['DBMSTheory', 'JavaTheory', 'PythonRTheory', 'StatsTheory', 'DataVizTheory', 
                   'BigDataTheory', 'LinuxCloudTheory', 'MLTheory']
lab_cols = ['DBMSLab', 'JavaLab', 'PythonRLab', 'StatsLab', 'DataVizLab', 'BigDataLab', 
                'LinuxCloudLab', 'MLLab']
    
percent_cols = ['10th_percentage', 'grad_percentage', 'Higher_Edu_Percent']
    
one_hot_col = ['branch_cleaned']


preprocessor = ColumnTransformer([
        ('num_imputer', SimpleImputer(strategy='median'), num_cols),  # Median imputation for numeric cols
        ('cat_imputer', SimpleImputer(strategy='most_frequent'), cat_cols),  # Mode imputation for categorical cols
  # One-hot encode a single column
    ], remainder='passthrough')



def preprocessing(X_df):
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

    grade_cols = ['Grade', 'Result', 'aptigrade', 'projectgrade']
    grade_mapping = {'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0, 'Fail': 0, 'Pass': 1}

    for col in grade_cols:
        if col in X_df.columns:
            X_df[col] = X_df[col].map(grade_mapping)

    return X_df

X_df = preprocessing(X_df)

df_merged = pd.concat([X_df, y.to_frame()], axis=1)
df = df_merged


# Define S3 bucket and file path
bucket_name = "glueoutbucket"
file_key = "data/dbda/previous/prev_processed.csv"

# Initialize S3 client
s3 = boto3.client("s3")

# Function to check if the file exists in S3
def check_file_exists(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False

if check_file_exists(bucket_name, file_key):
    # Read existing CSV file from S3
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    df2 = pd.read_csv(response['Body'])  # Read the CSV into df2

    # Merge df with df2
    df = pd.concat([df_merged, df2], ignore_index=True)

# Save the merged DataFrame back to S3
csv_buffer = StringIO()
df.to_csv(csv_buffer, index=False)

# Upload back to S3
s3.put_object(Bucket=bucket_name, Key=file_key, Body=csv_buffer.getvalue())



df = pd.read_csv('s3://glueoutbucket/data/dbda/previous/prev_processed.csv')

X = df.drop(['Is_Placed'],axis=1)
y = df['Is_Placed']



# Split data into train and test sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Increase dataset size to 600 (double the original size)
X_train_resampled, y_train_resampled = resample(X_train, y_train, replace=True, n_samples=1000, random_state=42)




# Initialize and train Random Forest model 
rf_model = RandomForestClassifier(bootstrap=True, max_depth=20, min_samples_leaf=1, 
                                  min_samples_split=5, n_estimators=100)
rf_model.fit(X_train_resampled, y_train_resampled)




# Predict on test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}\n")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))










