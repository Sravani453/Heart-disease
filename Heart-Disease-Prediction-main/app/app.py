from flask import Flask, request, render_template
import pickle
import numpy as np
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the pre-trained model with error handling
try:
    app.logger.debug("Attempting to load the model...")
    with open('heart_model.pkl', 'rb') as file:
        model = pickle.load(file)
    app.logger.debug("Model loaded successfully.")
except FileNotFoundError:
    app.logger.error("The model file 'heart_model.pkl' was not found.")
    raise FileNotFoundError("The model file 'heart_model.pkl' was not found. Please ensure it is in the project directory.")
except pickle.UnpicklingError:
    app.logger.error("The file 'heart_model.pkl' is not a valid pickle file or is corrupted.")
    raise ValueError("The file 'heart_model.pkl' is not a valid pickle file or is corrupted.")
except Exception as e:
    app.logger.error(f"An unexpected error occurred: {e}")
    raise

@app.route('/')
def home():
    """
    Renders the home page with the input form.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the form submission, processes input data, 
    and returns the prediction result.
    """
    if request.method == 'POST':
        try:
            # Extract input data from the form
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])

            # Validate input ranges
            if age <= 0 or sex not in [0, 1] or cp not in range(4) or trestbps <= 0 or chol <= 0:
                raise ValueError("Input values are out of valid ranges.")
        except ValueError as e:
            app.logger.error(f"Invalid input: {e}")
            return render_template('result.html', prediction="Invalid input. Please check the values and try again.")

        # Prepare the input for prediction
        input_data = np.array([[age, sex, cp, trestbps, chol]])

        # Make a prediction
        try:
            app.logger.debug(f"Making a prediction with input: {input_data}")
            prediction = model.predict(input_data)[0]
        except Exception as e:
            app.logger.error(f"Prediction error: {e}")
            return render_template('result.html', prediction=f"An error occurred during prediction: {str(e)}")

        # Generate a human-readable result
        if prediction == 1:
            result = "The patient is likely to have heart disease."
        else:
            result = "The patient is unlikely to have heart disease."

        # Render the result page
        app.logger.debug(f"Prediction result: {result}")
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
