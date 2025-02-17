import pandas as pd
import boto3
import warnings
from io import StringIO
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib  # or you can use pickle
from io import BytesIO

warnings.filterwarnings('ignore')

bucket_name = "glueoutbucket"
folder_path = "data/dac/previous/"

s3 = boto3.client("s3")

s3.put_object(Bucket=bucket_name, Key=folder_path)

print(f"Folder '{folder_path}' created successfully in bucket '{bucket_name}'.")


path = 's3://glueoutbucket/data/dac/dac_processed.csv'

def data_ingestion(path):
    drop_columns = [
        'form_number', 'PRN', 'dob', 'course',
        'pg_percentage', 'pre_ccat', 'Company_Name', 'Package', 'month', 'year', 
        'Age', 'Total800', 
        'OS_Total', 'CPP_Total', 'Java_Total', 'DSA_Total', 'DBT_Total', 
        'WPT_Total', 'WB_Java_Total', 'DotNet_Total', 'SDLC_Total',
        'OS_Status', 'CPP_Status', 'Java_Status', 'DSA_Status', 'DBT_Status',
        'WPT_Status', 'WB_Java_Status', 'DotNet_Status', 'SDLC_Status',
        'CCATrank'
    ]
    df = pd.read_csv(path)
    df = df.drop(drop_columns,axis=1)
    X_df = df.drop(['Is_Placed'],axis=1)
    y = df['Is_Placed']
    return X_df, y
X_df,y = data_ingestion(path)

cat_cols= ['Grade', 'Result', 'Apti_EC_Grade', 'Project_Grade', 'branch_cleaned']
theory_cols = ['OS_Theory','CPP_Theory','Java_Theory','DSA_Theory','DBT_Theory','WPT_Theory', 'WB_Java_Theory','DotNet_Theory','SDLC_Theory']
lab_cols = ['OS_Lab','CPP_Lab','Java_Lab','DSA_Lab','DBT_Lab','WPT_Lab','WB_Java_Lab','DotNet_Lab', 'SDLC_Lab']
percent_cols = ['10th_percentage', 'grad_percentage', 'Higher_Edu_Percent']
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

preprocessor = ColumnTransformer([
        ('num_imputer', SimpleImputer(strategy='median'), num_cols),  # Median imputation for numeric cols
        ('cat_imputer', SimpleImputer(strategy='most_frequent'), cat_cols),  # Mode imputation for categorical cols
        # ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), one_hot_col)  # One-hot encode a single column
    ], remainder='passthrough')

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

    grade_cols = ['Grade', 'Result', 'Apti_EC_Grade', 'Project_Grade']
    grade_mapping = {'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0, 'Fail': 0, 'Pass': 1}

    for col in grade_cols:
        if col in X_df.columns:
            X_df[col] = X_df[col].map(grade_mapping)

    return X_df

df_merged = pd.concat([X_df, y.to_frame()], axis=1)
df = df_merged

# Define S3 bucket and file path
bucket_name = "glueoutbucket"
file_key = "data/dac/previous/prev_processed.csv"

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
    df = pd.concat([df, df2], ignore_index=True)


# Save the merged DataFrame back to S3
csv_buffer = StringIO()
df.to_csv(csv_buffer, index=False)

# Upload back to S3
s3.put_object(Bucket=bucket_name, Key=file_key, Body=csv_buffer.getvalue())

print(f"File '{file_key}' updated successfully in bucket '{bucket_name}'.")


response = s3.get_object(Bucket=bucket_name, Key="data/dac/previous/prev_processed.csv" )

# Read the CSV file directly from the S3 response
csv_content = response['Body'].read().decode('utf-8')  # Decode from bytes to string
df = pd.read_csv(StringIO(csv_content)) 


X = df.drop(['Is_Placed'],axis=1)
y = df['Is_Placed']


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Print training completion
print("Random Forest model training completed!")

# Save the processor object directly to S3
processor_stream = BytesIO()
joblib.dump(preprocessor, processor_stream)
processor_stream.seek(0)

# Upload the processor to S3
s3 = boto3.client('s3')
bucket_name = 'glueoutbucket'
s3_processor_path = 'models/dac/latest_processor.pkl'

# Upload the processor to S3 directly from the byte stream
s3.put_object(Bucket=bucket_name, Key=s3_processor_path, Body=processor_stream)

# Save the model directly to S3
model_stream = BytesIO()
joblib.dump(rf_model, model_stream)
model_stream.seek(0)

s3_model_path = 'models/dac/latest_model.pkl'
s3.put_object(Bucket=bucket_name, Key=s3_model_path, Body=model_stream)

print(f"Processor uploaded to S3 at s3://{bucket_name}/{s3_processor_path}")
print(f"Model uploaded to S3 at s3://{bucket_name}/{s3_model_path}")



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
