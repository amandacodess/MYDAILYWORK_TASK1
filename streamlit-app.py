import streamlit as st
import pickle
import pandas as pd
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# Custom CSS for blue-black gradient background with white text
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #000428 0%, #004e92 100%);
    }
    
    /* All text white */
    .stApp, .stMarkdown, .stText, p, span, label, h1, h2, h3 {
        color: white !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #000428 0%, #003366 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Blue-Black Gradient buttons */
    .stButton > button {
        background: linear-gradient(135deg, #004e92 0%, #000428 100%) !important;
        color: white !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        padding: 12px 30px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 78, 146, 0.5) !important;
        font-size: 16px !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #000428 0%, #004e92 100%) !important;
        transform: scale(1.05);
        box-shadow: 0 6px 25px rgba(0, 78, 146, 0.8) !important;
        border: 2px solid rgba(255, 255, 255, 0.6) !important;
    }
    
    /* White slider */
    .stSlider > div > div > div {
        background-color: white !important;
    }
    
    .stSlider > div > div > div > div {
        background-color: white !important;
    }
    
    /* Slider thumb */
    .stSlider [role="slider"] {
        background-color: white !important;
        border: 2px solid white !important;
    }
    
    /* Slider track */
    .stSlider > div > div > div {
        background: rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Input boxes */
    .stSelectbox > div > div, .stNumberInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Dropdown text */
    .stSelectbox [data-baseweb="select"] > div {
        color: white !important;
    }
    
    /* Number input */
    .stNumberInput input {
        color: white !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: white !important;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: white !important;
    }
    
    .stAlert * {
        color: white !important;
    }
    
    /* Success/Error boxes */
    .stSuccess, .stError {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }
    
    .streamlit-expanderContent {
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Column borders */
    [data-testid="column"] {
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🚢 Titanic Survival Prediction")
st.markdown("**Predict whether a passenger would have survived the Titanic disaster**")
st.markdown("---")

# Load the trained model
@st.cache_resource
def load_model():
    with open('models/titanic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Sidebar - Project Info
st.sidebar.header("📊 Project Info")
st.sidebar.info("""
**Task 1: Titanic Survival Prediction**

- Model: Random Forest
- Accuracy: 82%
- Dataset: Kaggle Titanic
- Author: Amanda Caroline Young
""")

st.sidebar.markdown("---")
st.sidebar.header("🔗 Links")
st.sidebar.markdown("[GitHub Repository](https://github.com/YOUR_USERNAME/MYDAILYWORK_Task1)")

# Main content - Two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎯 Make a Prediction")
    
    # Input form
    st.subheader("Enter Passenger Details:")
    
    # Row 1
    c1, c2, c3 = st.columns(3)
    
    with c1:
        pclass = st.selectbox(
            "Passenger Class",
            options=[1, 2, 3],
            format_func=lambda x: f"Class {x} ({'1st' if x==1 else '2nd' if x==2 else '3rd'})"
        )
    
    with c2:
        sex = st.selectbox(
            "Gender",
            options=["Female", "Male"]
        )
        sex_encoded = 1 if sex == "Female" else 0
    
    with c3:
        age = st.slider("Age", min_value=0, max_value=80, value=29)
    
    # Row 2
    c4, c5, c6 = st.columns(3)
    
    with c4:
        sibsp = st.number_input("Siblings/Spouses", min_value=0, max_value=8, value=0)
    
    with c5:
        parch = st.number_input("Parents/Children", min_value=0, max_value=6, value=0)
    
    with c6:
        fare = st.number_input("Fare (£)", min_value=0.0, max_value=500.0, value=50.0, step=5.0)
    
    # Row 3
    embarked = st.selectbox(
        "Port of Embarkation",
        options=["Southampton", "Cherbourg", "Queenstown"]
    )
    embarked_encoded = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[embarked]
    
    # Predict button
    st.markdown("---")
    if st.button("🔮 Predict Survival", type="primary", use_container_width=True):
        # Create input dataframe
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex_encoded],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'Embarked': [embarked_encoded]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Display result
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        if prediction == 1:
            st.success("✅ **SURVIVED** - This passenger likely would have survived!")
            st.metric("Survival Probability", f"{probability[1]*100:.1f}%")
        else:
            st.error("❌ **DID NOT SURVIVE** - This passenger likely would not have survived.")
            st.metric("Survival Probability", f"{probability[1]*100:.1f}%")
        
        # Explanation
        st.info(f"""
        **Why this prediction?**
        
        Based on historical data:
        - {'Women' if sex == 'Female' else 'Men'} had {74 if sex == 'Female' else 19}% survival rate
        - Class {pclass} passengers had {63 if pclass==1 else 47 if pclass==2 else 24}% survival rate
        - Age {age} falls in the {'child' if age < 16 else 'adult' if age < 60 else 'elderly'} category
        """)

with col2:
    st.header("📈 Visualizations")
    
    # Show visualizations if they exist
    try:
        st.subheader("EDA Overview")
        eda_img = Image.open('visualizations/titanic_eda.png')
        st.image(eda_img, use_container_width=True)
    except:
        st.warning("EDA visualization not found")
    
    try:
        st.subheader("Model Performance")
        cm_img = Image.open('visualizations/confusion_matrix.png')
        st.image(cm_img, use_container_width=True)
    except:
        st.warning("Confusion matrix not found")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='color: white;'>Built with ❤️ using Streamlit | Data Science Internship Task 1</p>
    <p style='color: white;'><b>MyDailyWork Internship</b> | 2026</p>
</div>
""", unsafe_allow_html=True)

# Expandable section - Key Insights
with st.expander("💡 Key Insights from the Data"):
    st.markdown("""
    ### Main Findings:
    
    1. **Gender was the strongest predictor**
       - Women: 74% survival rate
       - Men: 19% survival rate
    
    2. **Socioeconomic status mattered**
       - 1st Class: 63% survival
       - 2nd Class: 47% survival
       - 3rd Class: 24% survival
    
    3. **Age played a role**
       - Children had higher survival rates
       - Elderly passengers had lower rates
    
    4. **"Women and children first" protocol**
       - Clearly visible in survival patterns
       - Social norms of the era evident in data
    """)

# Expandable section - About the Model
with st.expander("🤖 About the Model"):
    st.markdown("""
    ### Model Details:
    
    - **Algorithm:** Random Forest Classifier
    - **Test Accuracy:** 82%
    - **Cross-Validation Score:** 81%
    - **Training Dataset:** 891 passengers
    
    ### Features Used:
    1. Passenger Class (Pclass)
    2. Gender (Sex)
    3. Age
    4. Siblings/Spouses (SibSp)
    5. Parents/Children (Parch)
    6. Fare
    7. Port of Embarkation (Embarked)
    
    ### Top 3 Important Features:
    1. Sex (31%)
    2. Fare (26%)
    3. Age (23%)
    """)