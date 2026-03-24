import streamlit as st
import requests

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. CUSTOM CSS FOR STYLING
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        height: 50px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF2E2E;
        border-color: #FF2E2E;
    }
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# 3. SIDEBAR CONFIGURATION
with st.sidebar:
    st.title("🔬 About the App")
    st.info(
        """
        This machine learning app predicts whether a breast tumor is
        **malignant** or **benign** based on 30 numeric features computed
        from a digitized image of a fine needle aspirate (FNA) of a breast mass.
        
        The model uses the **Breast Cancer Wisconsin (Diagnostic)** dataset
        with a Random Forest classifier.
        """
    )
    st.write("---")
    st.caption("Built with Streamlit & Cloud Run")

# 4. MAIN APP INTERFACE
st.title("🔬 Breast Cancer Diagnosis Predictor")
st.markdown("Adjust the sliders below to input the tumor measurements.")

# Feature definitions with (label, min, max, default) for each group
mean_features = [
    ("mean_radius", "Radius", 6.0, 30.0, 14.1),
    ("mean_texture", "Texture", 9.0, 40.0, 19.3),
    ("mean_perimeter", "Perimeter", 40.0, 190.0, 92.0),
    ("mean_area", "Area", 140.0, 2600.0, 655.0),
    ("mean_smoothness", "Smoothness", 0.05, 0.17, 0.10),
    ("mean_compactness", "Compactness", 0.02, 0.35, 0.10),
    ("mean_concavity", "Concavity", 0.0, 0.45, 0.09),
    ("mean_concave_points", "Concave Points", 0.0, 0.2, 0.05),
    ("mean_symmetry", "Symmetry", 0.1, 0.3, 0.18),
    ("mean_fractal_dimension", "Fractal Dimension", 0.05, 0.1, 0.06),
]

error_features = [
    ("radius_error", "Radius", 0.1, 3.0, 0.4),
    ("texture_error", "Texture", 0.3, 5.0, 1.2),
    ("perimeter_error", "Perimeter", 0.7, 22.0, 2.9),
    ("area_error", "Area", 6.0, 550.0, 40.0),
    ("smoothness_error", "Smoothness", 0.001, 0.03, 0.007),
    ("compactness_error", "Compactness", 0.002, 0.14, 0.025),
    ("concavity_error", "Concavity", 0.0, 0.4, 0.03),
    ("concave_points_error", "Concave Points", 0.0, 0.05, 0.01),
    ("symmetry_error", "Symmetry", 0.007, 0.08, 0.02),
    ("fractal_dimension_error", "Fractal Dimension", 0.0, 0.03, 0.004),
]

worst_features = [
    ("worst_radius", "Radius", 7.0, 37.0, 16.3),
    ("worst_texture", "Texture", 12.0, 50.0, 25.7),
    ("worst_perimeter", "Perimeter", 50.0, 260.0, 107.0),
    ("worst_area", "Area", 185.0, 4300.0, 881.0),
    ("worst_smoothness", "Smoothness", 0.07, 0.23, 0.13),
    ("worst_compactness", "Compactness", 0.02, 1.1, 0.25),
    ("worst_concavity", "Concavity", 0.0, 1.3, 0.27),
    ("worst_concave_points", "Concave Points", 0.0, 0.3, 0.11),
    ("worst_symmetry", "Symmetry", 0.15, 0.66, 0.29),
    ("worst_fractal_dimension", "Fractal Dimension", 0.05, 0.21, 0.08),
]

data = {}

# Mean features
st.subheader("📊 Mean Values")
col1, col2 = st.columns(2)
for i, (key, label, mn, mx, default) in enumerate(mean_features):
    with col1 if i % 2 == 0 else col2:
        data[key] = st.slider(f"Mean {label}", mn, mx, default, key=key)

# Standard error features
st.subheader("📉 Standard Error Values")
col1, col2 = st.columns(2)
for i, (key, label, mn, mx, default) in enumerate(error_features):
    with col1 if i % 2 == 0 else col2:
        data[key] = st.slider(f"SE {label}", mn, mx, default, key=key)

# Worst features
st.subheader("📈 Worst (Largest) Values")
col1, col2 = st.columns(2)
for i, (key, label, mn, mx, default) in enumerate(worst_features):
    with col1 if i % 2 == 0 else col2:
        data[key] = st.slider(f"Worst {label}", mn, mx, default, key=key)

st.write("---")

# 5. PREDICTION LOGIC
if st.button('🔍 Predict Diagnosis'):

    with st.spinner('Analyzing tumor data...'):
        try:
            # API Call — update this URL after deploying to Cloud Run
            response = requests.post(
                'https://cancer-app-XXXXXXXXXX.us-east1.run.app/predict',
                json=data
            )

            if response.status_code == 200:
                prediction = response.json()['prediction']

                # Dynamic Result Display
                st.success("Prediction Complete!")

                res_col1, res_col2 = st.columns([1, 2])

                with res_col1:
                    if prediction == "benign":
                        st.markdown("## ✅")
                    else:
                        st.markdown("## ⚠️")

                with res_col2:
                    if prediction == "benign":
                        st.header("Prediction: **Benign**")
                        st.markdown("The model predicts the tumor is **benign** (non-cancerous).")
                    else:
                        st.header("Prediction: **Malignant**")
                        st.markdown("The model predicts the tumor is **malignant** (cancerous).")

                    st.caption("⚠️ This is a demo model — not for medical use.")
                    st.balloons()

            else:
                st.error(f'Server Error: {response.status_code}')

        except requests.exceptions.RequestException as e:
            st.error('Connection Error: Could not reach the prediction service.')
            