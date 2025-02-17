import streamlit as st
from PIL import Image
import os
import base64


# Load and display SM VITA logo
logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
if os.path.exists(logo_path):
    with open(logo_path, "rb") as img_file:
        logo_base64 = base64.b64encode(img_file.read()).decode()
else:
    st.warning("⚠️ Warning: Logo file VITA_logo.png not found! Upload it in the same directory.")
    logo_base64 = ""

st.markdown(
    f"""
    <div style="display: flex; align-items: center; text-align: center;">
        <img src="data:image/png;base64,{logo_base64}" width="150">
        <h2 style="margin: 0; font-size: 50px; font-weight: bold;">SMVITA</h2>
    </div>
    <p style="text-align: left; font-size: 20px;"> Smart CDAC Placement Forecasting with Machine Learning.</p>
    """, unsafe_allow_html=True
)

# Project Description
st.markdown("""
    ## About the Project  
    The **Placement Prediction System** is developed for **SM VITA** to help students
    and faculty analyze placement probabilities based on academic and technical skills.
    This system leverages **Machine Learning & Data Analytics** to provide insights into students' placement opportunities.
    Developed a predictive analytics platform using machine learning models to forecast placement outcomes for students based on their profiles. The solution leverages cloud technologies for scalability and provides actionable predictions to enhance decision-making for both institutions and students.
""")

# About Us - Team Section
st.markdown("""
    ## About Us  
    We are a team of passionate developers dedicated to building data-driven solutions.
    Our goal is to enhance decision-making through predictive analytics.
    We strive to bridge the gap between data and real-world applications, ensuring impactful results.
    Our expertise spans across multiple domains, allowing us to create robust and scalable solutions.


    **Team Members:**
    - **Dhanesh Nair** 
    - **Ketki Dandgavale** 
    - **Pritam Singh Rathore** 
    - **Priyank Acharekar** 
    - **Ram Jaybhaye** 
    - **Taroon Sharma** 
    - **Vedant Pednekar** 
    - **Vipul Patidar** 
""")

# Initialize session state variables if they don’t exist
if "show_smvita" not in st.session_state:
    st.session_state.show_smvita = False

if "show_features" not in st.session_state:
    st.session_state.show_features = False

# Toggle button for SM VITA details
if st.button("Learn More About SM VITA"):
    st.session_state.show_smvita = not st.session_state.show_smvita

# Display or hide SM VITA details
if st.session_state.show_smvita:
    st.markdown("[Visit SM VITA](https://www.vidyanidhi.com/)")

# Toggle button for System Features
if st.button("Explore the System Features"):
    st.session_state.show_features = not st.session_state.show_features

# Display or hide Features section
if st.session_state.show_features:
    st.write(
        """
        ### 🔹 Key Features of the System
        - 📊 **Accurate Placement Prediction:** Utilizes machine learning to forecast placement eligibility based on academic and demographic data.
        - ⚙️ **Automation of Placement Processes:** Automates data collection, preprocessing, and prediction tasks.
        - 📈 **Actionable Insights:** Provides interactive dashboards for stakeholders to make informed decisions.
        - ☁️ **Cloud Scalability:** Deploys the solution on the cloud for high availability and scalability.
        - 🎯 **Personalized Student Guidance:** Helps students identify areas for improvement based on predictive insights.
        """,
        unsafe_allow_html=True
    )

# Styling Enhancements
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        h1 {
            color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)
