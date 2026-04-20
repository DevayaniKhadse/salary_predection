import streamlit as st
import pandas as pd
import joblib

# Load the trained model, scaler, and label encoders
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

def predict_salary(input_data):
    # Create a DataFrame from input data
    df_input = pd.DataFrame([input_data])

    # Apply label encoding using the loaded encoders
    for column, encoder in label_encoders.items():
        if column in df_input.columns:
            # Get known classes from the encoder
            known_classes = encoder.classes_
            # Map input values to their encoded integers
            # If an input value is not in known_classes, replace it with -1 (or handle as appropriate)
            df_input[column] = df_input[column].apply(lambda x: encoder.transform([x])[0] if x in known_classes else -1)

    # Define the order of columns as expected by the model (excluding 'Salary')
    # These are the columns from the X DataFrame used for training
    expected_columns = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']

    # Ensure all expected columns are present, adding any missing with a default value
    for col in expected_columns:
        if col not in df_input.columns:
            df_input[col] = 0.0 # Default value, ideally derived from training data stats or mean

    # Reorder columns to match training data order
    df_input = df_input[expected_columns]

    # Scale numerical features using the loaded scaler
    scaled_input = scaler.transform(df_input)

    # Make prediction
    prediction = model.predict(scaled_input)
    return prediction[0]

st.title('Salary Prediction App')

st.write('Enter the details below to predict the salary.')

# Input fields for features, matching the original training data
age = st.slider('Age', 18.0, 65.0, 30.0, 0.5)

# For categorical columns, retrieve the original classes from the label encoders
gender_options = list(label_encoders['Gender'].classes_)
gender = st.selectbox('Gender', gender_options)

education_options = list(label_encoders['Education Level'].classes_)
education_level = st.selectbox('Education Level', education_options)

job_title_options = list(label_encoders['Job Title'].classes_)
job_title = st.selectbox('Job Title', job_title_options)

years_experience = st.slider('Years of Experience', 0.0, 40.0, 5.0, 0.5)


if st.button('Predict Salary'):
    input_data = {
        'Age': age,
        'Gender': gender,
        'Education Level': education_level,
        'Job Title': job_title,
        'Years of Experience': years_experience
    }

    try:
        predicted_salary = predict_salary(input_data)
        st.success(f'The predicted salary is: ₹{predicted_salary:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
