import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Restaurant Intelligence App", layout="wide")

st.title("Restaurant Intelligence Platform")
st.markdown("Analyze, Predict, and Visualize restaurant data using AI and maps.")

# Load models and vectorizers once at the top
@st.cache_resource
def load_models():
    cuisine_model = joblib.load("models/cuisine_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    
    rating_model = joblib.load("models/rating_model.pkl")

    df_reco = joblib.load("models/dataframe.pkl")
    
    # Load cosine matrix
    cosine_matrix = joblib.load("models/cosine_matrix.pkl")
    
    return cuisine_model, vectorizer, label_encoder, rating_model, df_reco, cosine_matrix

cuisine_model, vectorizer, label_encoder, rating_model, df_reco, cosine_matrix = load_models()
tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")

# Tabs
tabs = st.tabs([
    "Cuisine Classification", 
    "Rating Prediction", 
    "Recommendation System", 
    "Interactive Map"
])

# ---------------- TAB 1 ----------------
with tabs[0]:
    st.header("Cuisine Classification")
    st.write("Predict the cuisine type based on restaurant information.")

    name = st.text_input("Restaurant Name")
    city = st.text_input("City")
    address = st.text_input("Address")
    locality = st.text_input("Locality")
    locality_verbose = st.text_input("Locality Verbose")
    price_range = st.selectbox("Price Range", options=[1, 2, 3, 4])
    votes = st.text_input("Votes")

        
    if st.button("Predict Cuisine"):
        from scipy.sparse import hstack, csc_matrix
        input_text = f"{name} {locality} {city} {locality_verbose} {address}"
        input_vec = vectorizer.transform([input_text])

            #  Adding numeric features like training
        price_range_val = price_range
        votes_val = float(votes) if votes else 0
        numeric_features = csc_matrix([[price_range_val, votes_val]])

        final_input = hstack([input_vec, numeric_features])

        prediction = cuisine_model.predict(final_input)
        cuisine = label_encoder.inverse_transform(prediction)[0]
        st.success(f"**Predicted Cuisine: {cuisine.title()}**")


# ---------------- TAB 2 ----------------
with tabs[1]:
    st.header("Average Rating Prediction")
    st.write("Predict the expected rating of a restaurant.")

    name2 = st.text_input("Restaurant Name", key="rating_name")
    cuisines2 = st.text_input("Cuisines", key="rating_cuisine")
    city2 = st.text_input("City", key="rating_city")
    votes2 = st.text_input("Votes", key="rating_votes")

    if st.button("Predict Rating"):
        try:
            input_data = pd.DataFrame([{
                'Restaurant Name': name2,
                'Cuisines': cuisines2,
                'City': city2,
                'Votes': float(votes2) if votes2 else 0
            }])
            prediction = rating_model.predict(input_data)[0]
            st.success(f"**Predicted Rating: {round(prediction, 2)} / 5.0**")
        except Exception as e:
            st.error(f"Error predicting rating: {e}")

# ---------------- TAB 3 ----------------
with tabs[2]:
    st.header("Restaurant Recommendation System")
    st.write("Get top 5 similar restaurants.")

    selected_restaurant = st.selectbox("Select a Restaurant", df_reco['Restaurant Name'].unique())
    
    def get_recommendations_from_input(user_input, df_data, tfidf_vectorizer, cosine_matrix, tfidf_matrix, num_results=5):
    # Clean and vectorize user input
        from re import sub
        cleaned_input = sub(r'[^a-zA-Z0-9\s]', '', user_input.lower())
        user_vec = tfidf_vectorizer.transform([cleaned_input])
        
        # Compute cosine similarity with entire dataset
        similarity_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
        
        # Get top matches
        top_indices = similarity_scores.argsort()[-num_results:][::-1]
        recommendations = df_data.iloc[top_indices].copy()
        recommendations["Similarity Score"] = similarity_scores[top_indices]
        
        return recommendations[['Restaurant Name', 'Cuisines', 'City', 'Aggregate rating', 'Votes', 'Similarity Score']]
    
    user_input = st.text_input("Describe what you're looking for (e.g. 'cheap italian in delhi')")

    if st.button("Recommend"):
        if user_input:
            results = get_recommendations_from_input(user_input, df_reco, vectorizer, cosine_matrix, tfidf_matrix, num_results=5)
            st.write("**Top Recommendations:**")
            for i, row in results.iterrows():
                st.markdown(f"- **{row['Restaurant Name']}** ({row['City']}) — *{row['Cuisines']}*, Rating: {row['Aggregate rating']}")
        else:
            st.warning("Please enter a restaurant name or preference.")

# ---------------- TAB 4 ----------------
with tabs[3]:
    st.header("Interactive Restaurant Map")
    st.write("Explore restaurant locations on an interactive map.")

    map_path = os.path.join("maps", "Restaurant_Distribution_Clustered_Full.html")


    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            map_html = f.read()
            st.components.v1.html(map_html, height=600, scrolling=True)
    else:

        st.warning("Map file not found. Please ensure the HTML file is in the correct folder.")

