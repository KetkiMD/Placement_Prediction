import streamlit as st
import boto3
import json
import base64

# Load credentials from Streamlit secrets
VALID_USERNAME = st.secrets["APP_USERNAME"]
VALID_PASSWORD = st.secrets["APP_PASSWORD"]

AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_SESSION_TOKEN = st.secrets.get("AWS_SESSION_TOKEN", None)  # Optional token
AWS_REGION = st.secrets["AWS_REGION"]

# Authentication Function
def authenticate_user(username, password):
    return username == VALID_USERNAME and password == VALID_PASSWORD

# Login Page
def login():
    st.title("Login Page")

    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state["authenticated"] = True  # Removed `.sync()`
        else:
            st.error("Invalid Username or Password")

# AWS Lambda File Upload Function
def send_files_to_lambda(files, course_type, year, month):
    try:
        payload = {
            "course_type": course_type,
            "year": year,
            "month": month,
            "files": {key: None for key in files.keys()}
        }

        for key, file in files.items():
            if file:
                file_content = file.read()
                encoded_file = base64.b64encode(file_content).decode("utf-8")
                payload["files"][key] = encoded_file

        # Initialize AWS Lambda client with secrets
        lambda_client = boto3.client(
            'lambda',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            aws_session_token=AWS_SESSION_TOKEN,  # Optional
            region_name=AWS_REGION
        )

        response = lambda_client.invoke(
            FunctionName="merger",
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )

        response_payload = json.loads(response['Payload'].read())
        return response_payload

    except Exception as e:
        st.error(f"Error in send_files_to_lambda: {str(e)}")  # Logs error in Streamlit UI
        return {"statusCode": 500, "body": str(e)}

# Main Application
def main():
    # Ensure session state is initialized
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        login()
    else:
        st.title("Add Student records")

        course_type = st.selectbox("Select Course Type", ["DBDA", "DAC"])
        year = st.selectbox("Select a Year:", list(range(2010, 2030)))
        month = st.selectbox("Select a Month:", ["March", "May", "September"])

        files = {
            "result": st.file_uploader("Upload Result File", type=["csv", "xlsx"], key="result"),
            "placement": st.file_uploader("Upload Placement File", type=["csv", "xlsx"], key="placement"),
            "registration": st.file_uploader("Upload Registration File", type=["csv", "xlsx"], key="registration"),
            "master": st.file_uploader("Upload Master File", type=["csv", "xlsx"], key="master")
        }

        if st.button("Upload"):
            if all(files.values()):
                with st.spinner("Uploading files to AWS Lambda..."):
                    response = send_files_to_lambda(files, course_type, year, month)

                    if response["statusCode"] == 200:
                        st.success(f"Files uploaded successfully! Lambda Response: {response.get('message', 'No message')}")
                    else:
                        st.error(f"Error: {response.get('body', 'Unknown error')}")
            else:
                st.warning("Please upload all 4 required files before proceeding.")

        # Logout Button
        if st.button("Logout"):
            st.session_state["authenticated"] = False  

if __name__ == "__main__":
    main()
