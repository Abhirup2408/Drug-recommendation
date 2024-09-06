import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv(r'C:\Users\ACER\Downloads\drugsComTest_raw.csv\drugsComTest_raw.csv')

# Function to recommend drugs based on the condition
def recommend_drugs_for_condition(condition_input):
    # Select only the relevant columns
    df_relevant = df[['drugName', 'condition']]

    # Remove rows with missing conditions
    df_relevant.dropna(subset=['condition'], inplace=True)

    # Vectorize the 'condition' column using TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_relevant['condition'])

    # Transform the user input condition into a TF-IDF vector
    user_condition_tfidf = tfidf_vectorizer.transform([condition_input])

    # Compute cosine similarity between the user condition and all conditions in the dataset
    similarity_scores = cosine_similarity(user_condition_tfidf, tfidf_matrix)

    # Get the top 10 most similar conditions (highest similarity scores)
    top_indices = similarity_scores.argsort()[0][::-1][:10]

    # Retrieve the corresponding drug names
    top_medicines = df_relevant['drugName'].iloc[top_indices]

    # Return the top 10 recommended drugs as a list
    return top_medicines

# Streamlit UI
st.title("Medicine Recommendation System")

# Dropdown menu for condition selection
condition_options = [
    'Birth Control', 'Depression', 'Pain', 'Anxiety', 'Acne', 
    'Bipolar Disorder', 'Weight Loss', 'Insomnia', 'Obesity', 
    'ADHD', 'Emergency Contraception', 'Vaginal Yeast Infection', 
    'Diabetes Type 2', 'High Blood Pressure', 'Smoking Cessation'
]

selected_condition = st.selectbox("Select a health condition:", condition_options)

# Button to get recommendations
if st.button("Get Recommended Drugs"):
    if selected_condition:
        # Call the recommendation function
        recommended_drugs = recommend_drugs_for_condition(selected_condition)

        # Display the recommended drugs
        st.write(f"Top 10 recommended drugs for {selected_condition}:")
        for idx, drug in enumerate(recommended_drugs, 1):
            st.write(f"{idx}. {drug}")
