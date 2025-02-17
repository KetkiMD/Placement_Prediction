import streamlit as st
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dbda_model import predict_student_placement, calculate_total

st.title("Student Placement Prediction for DBDA")

# Select Course
course = "DBDA"

# Select Branch with predefined weights
branch_options = {
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
branch = st.selectbox("Select Branch", list(branch_options.keys()))

tenth_percentage = st.number_input("10th Percentage", min_value=0.0, max_value=100.0, value=50.0)
grad_percentage = st.number_input("Graduation Percentage", min_value=0.0, max_value=100.0, value=60.0)
higher_edu_percent = st.number_input("Higher Education Percentage", min_value=0.0, max_value=100.0, value=65.0)

grade = st.selectbox("Select Grade", ["A", "A+", "B", "C", "D", "F"])
aptigrade = st.selectbox("Aptitude Grade", ["A", "A+", "B", "C", "D", "F"])
projectgrade = st.selectbox("Project Grade", ["A", "A+", "B", "C", "D", "F"])

dbms_theory = st.number_input("DBMS Theory Marks", min_value=0, max_value=40, value=20)
dbms_lab = st.number_input("DBMS Lab Marks", min_value=0, max_value=60, value=40)
dbms_total = calculate_total(dbms_theory, dbms_lab)
st.write(f"DBMS Total Marks: {dbms_total}")

java_theory = st.number_input("Java Theory Marks", min_value=0, max_value=40, value=25)
java_lab = st.number_input("Java Lab Marks", min_value=0, max_value=60, value=48)
java_total = calculate_total(java_theory, java_lab)
st.write(f"Java Total Marks: {java_total}")

python_r_theory = st.number_input("Python & R Theory Marks", min_value=0, max_value=40, value=24)
python_r_lab = st.number_input("Python & R Lab Marks", min_value=0, max_value=60, value=47)
python_r_total = calculate_total(python_r_theory, python_r_lab)
st.write(f"Python & R Total Marks: {python_r_total}")

stats_theory = st.number_input("Statistics Theory Marks", min_value=0, max_value=40, value=36)
stats_lab = st.number_input("Statistics Lab Marks", min_value=0, max_value=60, value=50)
stats_total = calculate_total(stats_theory, stats_lab)
st.write(f"Statistics Total Marks: {stats_total}")

data_viz_theory = st.number_input("Data Visualization Theory Marks", min_value=0, max_value=40, value=36)
data_viz_lab = st.number_input("Data Visualization Lab Marks", min_value=0, max_value=60, value=30)
data_viz_total = calculate_total(data_viz_theory, data_viz_lab)
st.write(f"Data Visualization Total Marks: {data_viz_total}")

big_data_theory = st.number_input("Big Data Theory Marks", min_value=0, max_value=40, value=24)
big_data_lab = st.number_input("Big Data Lab Marks", min_value=0, max_value=60, value=32)

big_data_total = calculate_total(big_data_theory, big_data_lab)
st.write(f"Big Data Total Marks: {big_data_total}")

linux_cloud_theory = st.number_input("Linux & Cloud Theory Marks", min_value=0, max_value=40, value=31)
linux_cloud_lab = st.number_input("Linux & Cloud Lab Marks", min_value=0, max_value=60, value=30)
linux_cloud_total = calculate_total(linux_cloud_theory, linux_cloud_lab)
st.write(f"Linux & Cloud Total Marks: {linux_cloud_total}")

ml_theory = st.number_input("Machine Learning Theory Marks", min_value=0, max_value=40, value=30)
ml_lab = st.number_input("Machine Learning Lab Marks", min_value=0, max_value=60, value=33)
ml_total = calculate_total(ml_theory, ml_lab)
st.write(f"Machine Learning Total Marks: {ml_total}")

# Collecting all inputs
data = {
    '10th_percentage': tenth_percentage,
    'grad_percentage': grad_percentage,
    'DBMSTheory': dbms_theory,
    'DBMSLab': dbms_lab,
    'JavaTheory': java_theory,
    'JavaLab': java_lab,
    'PythonRTheory': python_r_theory,
    'PythonRLab':python_r_lab,
    'StatsTheory': stats_theory,
    'StatsLab': stats_lab,
    'DataVizTheory': data_viz_theory,
    'DataVizLab': data_viz_lab,
    'BigDataTheory': big_data_theory,
    'BigDataLab': big_data_lab,
    'LinuxCloudTheory':linux_cloud_theory,
    'LinuxCloudLab': linux_cloud_lab,
    'MLTheory': ml_theory,
    'MLLab': ml_lab,
    'Grade':grade,
    'aptigrade': aptigrade,
    'projectgrade': projectgrade,
    'Higher_Edu_Percent':higher_edu_percent,
    'branch_cleaned': branch
}
course="DBDA"
# Predict Placement
if st.button("Predict Placement"):
    prediction = predict_student_placement(data, course)
    
    placement_status = prediction['Placement Prediction']
    probability_yes = prediction['Probability Yes']
    probability_no = prediction['Probability No']
    
    st.success(f"**Predicted Placement:** {placement_status}")
    st.write(f"✅ **Probability of Getting Placed:** {probability_yes:.2f}%")
    st.write(f"❌ **Probability of Not Getting Placed:** {probability_no:.2f}%")

