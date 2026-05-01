import streamlit as st
import pandas as pd
import joblib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Salary Predictor", layout="wide")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
body {
    background: #f4f5fb;
}
.main {
    background: #f4f5fb;
}

/* Gradient Header */
.header {
    background: linear-gradient(135deg, #5f2eea, #7b61ff);
    padding: 30px;
    border-radius: 20px;
    color: white;
    text-align: center;
}

/* Card */
.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
}

/* Button */
.stButton>button {
    background: linear-gradient(135deg, #5f2eea, #7b61ff);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
}

/* Skill tags */
.skill {
    display:inline-block;
    background:#eee;
    padding:6px 12px;
    border-radius:20px;
    margin:5px;
}

/* Salary Card */
.salary-box {
    background: linear-gradient(135deg, #5f2eea, #7b61ff);
    color: white;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# -------------------- FUNCTION --------------------
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

# -------------------- HEADER --------------------
st.markdown("""
<div class="header">
    <h1>💼 Salary Prediction</h1>
    <p>Get AI-powered salary insights</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# -------------------- MAIN LAYOUT --------------------
col1, col2 = st.columns([2, 1])

# -------------------- INPUT FORM --------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Fill your details")

    age = st.slider('Age', 18.0, 65.0, 25.0)
    gender = st.selectbox('Gender', list(label_encoders['Gender'].classes_))
    education = st.selectbox('Education', list(label_encoders['Education Level'].classes_))
    job = st.selectbox('Job Role', list(label_encoders['Job Title'].classes_))
    exp = st.slider('Experience (Years)', 0.0, 40.0, 3.0)

    # Skills UI (NEW FEATURE)
    st.markdown("### Skills")
    skills = ["Python", "SQL", "Machine Learning", "Deep Learning", "Data Analysis"]
    selected_skills = st.multiselect("", skills)

    predict_btn = st.button("Predict Salary")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- OUTPUT --------------------
with col2:
    if predict_btn:
        input_data = {
            'Age': age,
            'Gender': gender,
            'Education Level': education,
            'Job Title': job,
            'Years of Experience': exp
        }

        salary = predict_salary(input_data)

        st.markdown(f"""
        <div class="salary-box">
            <h3>Estimated Salary</h3>
            <h1>₹{salary:,.0f}</h1>
            <p>💹 Based on your profile</p>
        </div>
        """, unsafe_allow_html=True)

        # -------------------- INSIGHTS (NEW FEATURE) --------------------
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Insights")

        if exp > 5:
            st.success("Experience is boosting your salary 📈")

        if "Machine Learning" in selected_skills:
            st.info("ML skill increases earning potential 🚀")

        if education.lower().startswith("master"):
            st.warning("Higher education gives advantage 🎓")

        st.markdown('</div>', unsafe_allow_html=True)

        # -------------------- SALARY RANGE (NEW FEATURE) --------------------
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Salary Range")

        low = salary * 0.8
        high = salary * 1.2

        st.write(f"₹{low:,.0f}  —  ₹{high:,.0f}")
        st.progress(70)

        st.markdown('</div>', unsafe_allow_html=True)
