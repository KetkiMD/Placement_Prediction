# Placement Prediction System

## Overview
The Placement Prediction System is a machine learning-based platform that forecasts student placement outcomes based on academic performance and other relevant factors. This project utilizes AWS services for data processing, model training, and inference, with a user-friendly Streamlit interface for data ingestion and prediction retrieval.

## Architecture Diagram![image](https://github.com/user-attachments/assets/60136428-2882-4bb6-a7b6-70f931d79312)


## Features
- **Data Ingestion:** Streamlit-based UI for authorized users to upload student data.
- **Data Processing:** AWS Lambda functions merge and store data in S3, triggering AWS Glue for transformation and cleaning.
- **Model Training:** Amazon SageMaker trains a binary classification model on cleaned data.
- **Prediction Interface:** Another Streamlit UI allows users to input academic details and receive placement predictions.

## Tech Stack
- **Frontend:** Streamlit
- **Backend:** AWS Lambda, AWS Glue, Amazon SageMaker
- **Data Storage:** AWS S3
- **Processing Engine:** Apache Spark (via AWS Glue)
- **Model Type:** Binary Classification (Placement Prediction)
- **Notebook Environment:** Jupyter

## Data Schema
| Feature                 | Description                           |
|-------------------------|---------------------------------------|
| `10th %`                | Percentage scored in 10th grade       |
| `12% or Diploma %`      | Percentage scored in 12th or diploma  |
| `Graduation %`          | Percentage scored in Graduation       |
| `Branch`                | Student's academic branch             |
| `Marks for  subjects`   | Subject-wise marks details            |
| `Grade`                 | Overall grade classification          |

## Workflow
1. **Data Ingestion:** Authorized users upload data via the Streamlit UI.
2. **Schema Transformation:** AWS Lambda processes and merges files, storing them in S3.
3. **Data Cleaning:** Another Lambda function triggers an AWS Glue job for data cleaning.
4. **Storage:** Cleaned data is stored in an S3 bucket.
5. **Model Training:** Amazon SageMaker trains a machine learning model on cleaned data.
6. **Prediction Interface:** Users input their marks in Streamlit to get placement predictions from the trained model.



## Future Enhancements
- Expand features for a multi-class classification approach.
- Incorporate additional student attributes such as extracurricular activities.
- Automate model retraining based on new data.

## Contributors
- **Dhanesh Nair**
- **Ketki Dandgavale**
- **Pritam Singh Rathore**
- **Priyank Acharekar**
- **Ram Jaybhaye**
- **Tarun Kumar**
- **Vedant Pednekar**
- **Vipul Patidar**


