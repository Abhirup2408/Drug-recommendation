import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import streamlit as st

# Load the dataset
data = pd.read_csv(r'drug_consumption.csv')

# Preprocessing the dataset
# Convert drug usage categorical levels (CL0, CL1, etc.) to numeric values

drug_columns = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 
                'Coke', 'Crack', 'Ecstacy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 
                'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']

# Create label encoders for the drugs
label_encoders = {}
for drug in drug_columns:
    le = LabelEncoder()
    data[drug] = le.fit_transform(data[drug])
    label_encoders[drug] = le

# Features for model training (demographic and personality traits)
features = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 
            'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']

X = data[features]

# Function to build, train, and predict for a given drug
def train_and_predict_drug(drug_name, X, data):
    # Split the data for the selected drug
    y = data[drug_name]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize the Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

# Function to make a prediction for a specific drug, input features, and current stock
def predict_drug_amount(model, drug_name, input_data, current_stock, target_category):
    # Predict drug usage for the input data
    prediction = model.predict([input_data])[0]
    
    # Decode the prediction back to the original CL category
    predicted_category = label_encoders[drug_name].inverse_transform([prediction])[0]
    
    # Define drug-specific thresholds for stock levels (these are example values, adjust as needed)
    drug_usage_thresholds = {
        'Alcohol': {'CL0': 0, 'CL1': 50, 'CL2': 100, 'CL3': 150, 'CL4': 200, 'CL5': 250, 'CL6': 300},
        'Amphet': {'CL0': 0, 'CL1': 10, 'CL2': 30, 'CL3': 60, 'CL4': 100, 'CL5': 150, 'CL6': 200},
        'Cannabis': {'CL0': 0, 'CL1': 40, 'CL2': 80, 'CL3': 120, 'CL4': 160, 'CL5': 200, 'CL6': 250},
        'Coke': {'CL0': 0, 'CL1': 10, 'CL2': 20, 'CL3': 40, 'CL4': 70, 'CL5': 100, 'CL6': 150},
        # Add specific thresholds for other drugs as needed
    }
    
    # Get the thresholds for the selected drug
    thresholds = drug_usage_thresholds.get(drug_name, drug_usage_thresholds['Alcohol'])  # Default to Alcohol if not specified
    
    # Determine the required stock based on the user-selected category
    required_stock = thresholds.get(target_category, 0)
    
    # Check if the current stock is below the required stock
    if current_stock < required_stock:
        amount_needed = required_stock - current_stock
        return f"{drug_name} usage category is {predicted_category}. You need to order {amount_needed} units to match {target_category}."
    else:
        return f"{drug_name} usage category is {predicted_category}. No immediate need to order. You have enough for {target_category}."

# Main function to handle user input via Streamlit
def main():
    st.title("Drug Consumption Prediction and Stock Management")

    # Select drug name from dropdown
    drug_name = st.selectbox("Select a drug:", drug_columns)
    
    # Select the target category (CL0, CL1, etc.) from dropdown
    target_category = st.selectbox("Select the target usage category (CL0 - CL6):", ['CL0', 'CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6'])
    
    # Input current stock level
    current_stock = st.number_input("Enter the current stock level:", min_value=0, step=1)
    
    # Train the model for the selected drug
    model = train_and_predict_drug(drug_name, X, data)
    
    # Example input data (This would be dynamic or from user inputs in real scenarios)
    input_data = [0.5, 0.5, 0.1, 0.9, 0.2, 0.3, 0.6, 0.4, 0.5, 0.7, 0.1, 0.2]
    
    # Predict for the selected drug and category
    if st.button("Predict Drug Requirement"):
        result = predict_drug_amount(model, drug_name, input_data, current_stock, target_category)
        st.write(result)

# Run the main function in Streamlit
if __name__ == "__main__":
    main()
