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

    selected_restaurant = st.text_input("Enter a Restaurant Name")
    
    def get_recommendations(name, df, cosine_sim):
        if name not in df['Restaurant Name'].values:
            return ["Restaurant not found."]
        idx = df[df['Restaurant Name'] == name].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        restaurant_indices = [i[0] for i in sim_scores]
        return df['Restaurant Name'].iloc[restaurant_indices].tolist()
    
    if st.button("Recommend"):
        recommendations = get_recommendations(selected_restaurant, df_reco, cosine_matrix)
        if "not found" in recommendations[0].lower():
            st.warning("Restaurant not found in database.")
        else:
            st.write("**Top 5 Similar Restaurants:**")
            for r in recommendations:
                st.markdown(f"- {r}")

# ---------------- TAB 4 ----------------
with tabs[3]:
    st.header("Interactive Restaurant Map")
    st.write("Explore restaurant locations on an interactive map.")

    map_path = map_path = r"C:/Users/Administrator/Downloads/Tasks/Restaurant_Distribution_Clustered_Full.html"


    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            map_html = f.read()
            st.components.v1.html(map_html, height=600, scrolling=True)
    else:
        st.warning("Map file not found. Please ensure the HTML file is in the correct folder.")