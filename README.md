Heart Disease Detection Model
This project demonstrates a machine learning model for detecting heart disease using a logistic regression algorithm. The model is built using Python and commonly used libraries such as pandas, numpy, and scikit-learn.

Dataset
The dataset used is named heart.csv, containing features related to heart health and a target variable indicating the presence (1) or absence (0) of heart disease.

Features:
Age
Sex
Chest pain type
Resting blood pressure
Serum cholesterol
Fasting blood sugar
Resting electrocardiographic results
Maximum heart rate achieved
Exercise-induced angina
ST depression induced by exercise
Slope of the peak exercise ST segment
Number of major vessels colored by fluoroscopy
Thalassemia (normal, fixed defect, reversible defect)


Target:
0: No heart disease
1: Heart disease detected

Project Structure
Heart_Disease_Detection.ipynb: The Jupyter notebook containing the code for loading the dataset, training the model, and evaluating its performance.
heart.csv: The dataset containing heart health records and outcomes.
README.md: This file, explaining the project structure and functionality.

Requirements
Python 3.x
Libraries:
pandas
numpy

scikit-learn
You can install the required libraries using the following command:


pip install -r requirements.txt

Steps to Run the Project
Clone the repository and navigate to the project directory.
Ensure you have the dataset heart.csv in the project folder.
Open the Jupyter notebook Heart_Disease_Detection.ipynb.
Run each cell sequentially to:
Load the dataset
Train the logistic regression model
Evaluate model performance
Make predictions
Model Training
The model is trained using the Logistic Regression algorithm, and the dataset is split into training and testing sets (80%-20%).


model = LogisticRegression()
model.fit(X_train, Y_train)
Model Evaluation
The model's accuracy on both the training and test sets is calculated to evaluate performance:

accuracy_score(Y_test, X_test_prediction)
Sample Prediction
The model can also predict heart disease for individual records:

input_data = X_test.iloc[18].values
prediction = model.predict(input_data.reshape(1, -1))
