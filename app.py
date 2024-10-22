from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the KNN and Naive Bayes models once when the app starts
knn_model = load("model/knn_model.pkl")
naive_bayes_model = load("model/naive_bayes_model.pkl")
decision_tree_model = load("model/decision_tree_model.pkl") # Load the Naive Bayes model
print("Loaded KNN model type:", type(knn_model))
print("Loaded Naive Bayes model type:", type(naive_bayes_model))
print("Loaded Decision tree model type:", type(decision_tree_model))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/classify", methods=['POST'])
def classify():
    try:
        # Get the classifier choice from the form
        classifier = request.form.get('classifier')  # Get the selected classifier
        print("Selected classifier:", classifier)  # Print for debugging

        # Access user inputs from the form
        clump = float(request.form.get('Clump_thickness', 0))
        uniformity_size = float(request.form.get('Uniformity_Cell_Size', 0))
        uniformity_shape = float(request.form.get('Uniformity_Cell_Shape', 0))
        adhesion = float(request.form.get('Marginal_Adhesion', 0))
        epithelial = float(request.form.get('Single_Epithelial_Cell_Size', 0))
        chromatin = float(request.form.get('Bland_Chromatin', 0))
        nucleoli = float(request.form.get('Normal_Nucleoli', 0))
        mitoses = float(request.form.get('Mitoses', 0))

        # Prepare input data for prediction
        input_data = pd.DataFrame([[clump, uniformity_size, uniformity_shape, adhesion, epithelial, chromatin, nucleoli, mitoses]],
                                  columns=['Clump_thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape', 
                                           'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bland_Chromatin', 
                                           'Normal_Nucleoli', 'Mitoses'])

        # Make the prediction based on the selected classifier
        if classifier == 'Naive Bayes':
            prediction = naive_bayes_model.predict(input_data)
        elif classifier == 'Nearest Neighbor':  # Default to KNN
            prediction = knn_model.predict(input_data)
        else:
            prediction = decision_tree_model.predict(input_data)

        # Determine the result based on the prediction
        if prediction[0] == 2:
            result = "Benign"
        elif prediction[0] == 4:
            result = "Malignant"
        else:
            result = "Unknown"

        # Collect user inputs for response
        user_input = {
            'Clump_thickness': clump,
            'Uniformity_Cell_Size': uniformity_size,
            'Uniformity_Cell_Shape': uniformity_shape,
            'Marginal_Adhesion': adhesion,
            'Single_Epithelial_Cell_Size': epithelial,
            'Bland_Chromatin': chromatin,
            'Normal_Nucleoli': nucleoli,
            'Mitoses': mitoses,
        }

        # Return the prediction, classifier used, and user input
        return jsonify({'prediction': result, 'classifier': classifier, 'user_input': user_input})

    except Exception as e:
        print("Error:", e)  # Print the error for debugging
        return jsonify({'error': str(e)}), 400
    
if __name__ == "__main__":
    app.run(debug=True)
