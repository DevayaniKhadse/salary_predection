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
        # Handle potential unseen labels by adding them to the encoder
        # (This is a simplified approach, in production, you'd handle unseen labels differently)
        # For example, by mapping them to an 'unknown' category or using a robust encoder
        for col_name in df_input.columns:
            if col_name == column and col_name in encoder.classes_:
                df_input[column] = df_input[column].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
            elif col_name == column and col_name not in encoder.classes_:
                # If the input value is not in the encoder's classes, try to append it temporarily
                # This is an imperfect solution for demonstration, production code needs a strategy for new categories
                try:
                    df_input[column] = encoder.transform(df_input[column])
                except ValueError:
                    # Fallback for truly unseen categories, assign a default or handle as unknown
                    st.warning(f"Unseen category encountered for {column}. Assigning -1 (or appropriate default).")
                    df_input[column] = -1 # Or a more robust handling, like the mode

    # Define the order of columns as expected by the model (excluding 'Salary')
    expected_columns = ['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles']

    # Ensure all expected columns are present, adding any missing with a default value
    for col in expected_columns:
        if col not in df_input.columns:
            df_input[col] = 0 # Default value, ideally derived from training data stats
    
    # Reorder columns to match training data order
    df_input = df_input[expected_columns]

    # Scale numerical features using the loaded scaler
    scaled_input = scaler.transform(df_input)

    # Make prediction
    prediction = model.predict(scaled_input)
    return prediction[0]

st.title('Salary Prediction App')

st.write('Enter the details below to predict the salary.')

# Input fields for features
rating = st.slider('Rating', 0.0, 5.0, 3.5, 0.1)
company_name_options = list(label_encoders['Company Name'].classes_)
company_name = st.selectbox('Company Name', company_name_options)
job_title_options = list(label_encoders['Job Title'].classes_)
job_title = st.selectbox('Job Title', job_title_options)
salaries_reported = st.number_input('Salaries Reported', min_value=1, value=5)
location_options = list(label_encoders['Location'].classes_)
location = st.selectbox('Location', location_options)
employment_status_options = list(label_encoders['Employment Status'].classes_)
employment_status = st.selectbox('Employment Status', employment_status_options)
job_roles_options = list(label_encoders['Job Roles'].classes_)
job_roles = st.selectbox('Job Roles', job_roles_options)


if st.button('Predict Salary'):
    input_data = {
        'Rating': rating,
        'Company Name': company_name,
        'Job Title': job_title,
        'Salaries Reported': salaries_reported,
        'Location': location,
        'Employment Status': employment_status,
        'Job Roles': job_roles
    }
    
    try:
        predicted_salary = predict_salary(input_data)
        st.success(f'The predicted salary is: ₹{predicted_salary:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
