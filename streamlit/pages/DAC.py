import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dac_model import predict_student_placement, calculate_total

st.title("Student Placement Prediction for DAC")

# Select Course
course = "DAC"

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

tenth_percentage = st.number_input("10th Percentage", min_value=0.0, max_value=100.0, value=94.0)
grad_percentage = st.number_input("Graduation Percentage", min_value=0.0, max_value=100.0, value=78.0)
higher_edu_percent = st.number_input("Higher Education Percentage", min_value=0.0, max_value=100.0, value=69.0)

grade = st.selectbox("Select Grade", ["A", "A+", "B", "C", "D", "F"], index=2)
apti_ec_grade = st.selectbox("Aptitude EC Grade", ["A", "A+", "B", "C", "D", "F"], index=2)
project_grade = st.selectbox("Project Grade", ["A", "A+", "B", "C", "D", "F"], index=2)

os_theory = st.number_input("OS Theory Marks", min_value=0, max_value=40, value=20)
os_lab = st.number_input("OS Lab Marks", min_value=0, max_value=60, value=20)
os_total = calculate_total(os_theory, os_lab)
st.write(f"OS Total Marks: {os_total}")

cpp_theory = st.number_input("C++ Theory Marks", min_value=0, max_value=40, value=20)
cpp_lab = st.number_input("C++ Lab Marks", min_value=0, max_value=60, value=25)
cpp_total = calculate_total(cpp_theory, cpp_lab)
st.write(f"C++ Total Marks: {cpp_total}")

java_theory = st.number_input("Java Theory Marks", min_value=0, max_value=40, value=20)
java_lab = st.number_input("Java Lab Marks", min_value=0, max_value=60, value=40)
java_total = calculate_total(java_theory, java_lab)
st.write(f"Java Total Marks: {java_total}")

dsa_theory = st.number_input("DSA Theory Marks", min_value=0, max_value=40, value=20)
dsa_lab = st.number_input("DSA Lab Marks", min_value=0, max_value=60, value=50)
dsa_total = calculate_total(dsa_theory, dsa_lab)
st.write(f"DSA Total Marks: {dsa_total}")

dbt_theory = st.number_input("DBT Theory Marks", min_value=0, max_value=40, value=20)
dbt_lab = st.number_input("DBT Lab Marks", min_value=0, max_value=60, value=45)
dbt_total = calculate_total(dbt_theory, dbt_lab)
st.write(f"DBT Total Marks: {dbt_total}")

wpt_theory = st.number_input("WPT Theory Marks", min_value=0, max_value=40, value=20)
wpt_lab = st.number_input("WPT Lab Marks", min_value=0, max_value=60, value=55)
wpt_total = calculate_total(wpt_theory, wpt_lab)
st.write(f"WPT Total Marks: {wpt_total}")

wb_java_theory = st.number_input("WB Java Theory Marks", min_value=0, max_value=40, value=20)
wb_java_lab = st.number_input("WB Java Lab Marks", min_value=0, max_value=60, value=60)
wb_java_total = calculate_total(wb_java_theory, wb_java_lab)
st.write(f"WB Java Total Marks: {wb_java_total}")

net_theory = st.number_input(".NET Theory Marks", min_value=0, max_value=40, value=20)
net_lab = st.number_input(".NET Lab Marks", min_value=0, max_value=60, value=40)
net_total = calculate_total(net_theory, net_lab)
st.write(f".NET Total Marks: {net_total}")

sdlc_theory = st.number_input("SDLC Theory Marks", min_value=0, max_value=40, value=20)
sdlc_lab = st.number_input("SDLC Lab Marks", min_value=0, max_value=60, value=55)
sdlc_total = calculate_total(sdlc_theory, sdlc_lab)
st.write(f"SDLC Total Marks: {sdlc_total}")

# Collecting all inputs
data = {
    '10th_percentage': tenth_percentage,
    'grad_percentage': grad_percentage,
    'OS_Theory': os_theory,
    'OS_Lab': os_lab,
    'CPP_Theory': cpp_theory,
    'CPP_Lab': cpp_lab,
    'Java_Theory': java_theory,
    'Java_Lab': java_lab,
    'DSA_Theory': dsa_theory,
    'DSA_Lab': dsa_lab,
    'DBT_Theory': dbt_theory,
    'DBT_Lab': dbt_lab,
    'WPT_Theory': wpt_theory,
    'WPT_Lab': wpt_lab,
    'WB_Java_Theory': wb_java_theory,
    'WB_Java_Lab': wb_java_lab,
    'DotNet_Theory': net_theory,
    'DotNet_Lab': net_lab,
    'SDLC_Theory': sdlc_theory,
    'SDLC_Lab': sdlc_lab,
    'Higher_Edu_Percent': higher_edu_percent,
    'Grade': grade,
    'Apti_EC_Grade': apti_ec_grade,
    'Project_Grade': project_grade,
    'branch_cleaned': branch
}

# Predict Placement
if st.button("Predict Placement"):
    prediction = predict_student_placement(data, course)
    
    placement_status = prediction['Placement Prediction']
    probability_yes = prediction['Probability Yes']
    probability_no = prediction['Probability No']
    
    st.success(f"**Predicted Placement:** {placement_status}")
    st.write(f"✅ **Probability of Getting Placed:** {probability_yes:.2f}%")
    st.write(f"❌ **Probability of Not Getting Placed:** {probability_no:.2f}%")

