import streamlit as st
import pandas as pd
import joblib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    page_icon="💼",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fb;
        }
        .card {
            padding: 20px;
            border-radius: 12px;
            background: white;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        }
        .metric-card {
            padding: 25px;
            border-radius: 12px;
            background: linear-gradient(135deg, #6C63FF, #8E7CFF);
            color: white;
            text-align: center;
        }
        .title {
            font-size: 28px;
            font-weight: 600;
        }
        .subtitle {
            color: gray;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# -------------------- FUNCTION (UNCHANGED) --------------------
def predict_salary(input_data):
    df_input = pd.DataFrame([input_data])

    for column, encoder in label_encoders.items():
        if column in df_input.columns:
            known_classes = encoder.classes_
            df_input[column] = df_input[column].apply(
                lambda x: encoder.transform([x])[0] if x in known_classes else -1
            )

    expected_columns = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']

    for col in expected_columns:
        if col not in df_input.columns:
            df_input[col] = 0.0

    df_input = df_input[expected_columns]

    scaled_input = scaler.transform(df_input)

    prediction = model.predict(scaled_input)
    return prediction[0]

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("💼 Salary AI")
    st.markdown("### Navigation")
    st.markdown("- Dashboard\n- Prediction\n- Insights")
    st.markdown("---")
    st.caption("Built for deployment")

# -------------------- HEADER --------------------
st.markdown('<div class="title">Salary Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered salary estimation system</div>', unsafe_allow_html=True)

st.markdown("")

# -------------------- INPUT SECTION --------------------
col1, col2 = st.columns([2, 1])

with col1:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Enter Candidate Details")

        c1, c2 = st.columns(2)

        with c1:
            age = st.slider('Age', 18.0, 65.0, 30.0, 0.5)

            gender_options = list(label_encoders['Gender'].classes_)
            gender = st.selectbox('Gender', gender_options)

            education_options = list(label_encoders['Education Level'].classes_)
            education_level = st.selectbox('Education Level', education_options)

        with c2:
            job_title_options = list(label_encoders['Job Title'].classes_)
            job_title = st.selectbox('Job Title', job_title_options)

            years_experience = st.slider('Years of Experience', 0.0, 40.0, 5.0, 0.5)

        predict_btn = st.button("🚀 Predict Salary")

        st.markdown('</div>', unsafe_allow_html=True)

# -------------------- OUTPUT SECTION --------------------
with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("💰 Predicted Salary")

    if predict_btn:
        input_data = {
            'Age': age,
            'Gender': gender,
            'Education Level': education_level,
            'Job Title': job_title,
            'Years of Experience': years_experience
        }

        try:
            predicted_salary = predict_salary(input_data)
            st.markdown(f"<h2>₹{predicted_salary:,.2f}</h2>", unsafe_allow_html=True)
            st.success("Prediction Successful")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.markdown("<h3>--</h3>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("© 2026 Salary Prediction System | Ready for deployment")
