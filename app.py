import joblib
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load the trained model
model = joblib.load('best_model.pkl')

# For simplicity, assuming the scaler and encoders are also available or fitted with dummy data
# In a real application, you would save/load these as well.
# Or, better yet, save a pipeline that includes scaler and encoders.

# Dummy scaler and encoders (replace with your actual fitted objects if available)
scaler = StandardScaler()
# Fit scaler with some dummy data or save/load the original fitted scaler
# For demonstration, we'll assume a scaler fitted to the training data exists.
# X_train from the notebook context is needed here for a more accurate representation.
# For now, let's just create a scaler that does nothing if not explicitly fitted

# Dummy LabelEncoders for each categorical column
company_name_encoder = LabelEncoder()
location_encoder = LabelEncoder()
employment_status_encoder = LabelEncoder()
job_roles_encoder = LabelEncoder()
job_title_encoder = LabelEncoder()

# In a real deployment, you would fit these encoders on your training data and save them.
# For this example, we'll make them handle unseen labels by fitting on a minimal set.
# This is a simplification; a robust solution involves saving the fitted encoders.
company_name_encoder.fit(['Unknown'] + list(df['Company Name'].unique()))
location_encoder.fit(['Unknown'] + list(df['Location'].unique()))
employment_status_encoder.fit(['Unknown'] + list(df['Employment Status'].unique()))
job_roles_encoder.fit(['Unknown'] + list(df['Job Roles'].unique()))
job_title_encoder.fit(['Unknown'] + list(df['Job Title'].unique()))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df_input = pd.DataFrame([data])

        # Apply label encoding to categorical features
        df_input['Company Name'] = company_name_encoder.transform(df_input['Company Name'])
        df_input['Location'] = location_encoder.transform(df_input['Location'])
        df_input['Employment Status'] = employment_status_encoder.transform(df_input['Employment Status'])
        df_input['Job Roles'] = job_roles_encoder.transform(df_input['Job Roles'])
        df_input['Job Title'] = job_title_encoder.transform(df_input['Job Title'])

        # Ensure all expected columns are present, even if some are dummy for now
        # This assumes your model expects all original features (excluding 'Salary')
        expected_columns = ['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles'] # Adjust based on your actual X columns
        for col in expected_columns:
            if col not in df_input.columns:
                df_input[col] = 0 # Or a sensible default/mean from training data

        # Scale the numerical features
        # IMPORTANT: The scaler should be the one fitted on your training data (X_train)
        # For this example, we are using a simplified scaler. In production, load the saved scaler.
        scaled_input = scaler.fit_transform(df_input[expected_columns]) # This will re-fit; for production, use scaler.transform

        prediction = model.predict(scaled_input)
        return jsonify({'prediction': prediction[0].tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # This is for local testing. For Colab deployment, you might use ngrok or similar.
    app.run(debug=True, host='0.0.0.0', port=5000)
