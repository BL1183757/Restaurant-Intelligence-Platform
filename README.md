# Restaurant Intelligence Platform

An AI-powered web application to analyze, predict, and visualize restaurant data using machine learning models and interactive maps. This project was built using Python, Streamlit, Scikit-learn, and other powerful libraries.

---

## Key Features

1. **Cuisine Classification**
   - Predict the type of cuisine offered by a restaurant based on text and numeric features.
   - Uses Gradient Boosting Classifier trained on restaurant metadata.

2. **Average Rating Prediction**
   - Predicts the average user rating (out of 5.0) for a restaurant using features like name, cuisine, city, and vote count.
   - Built with Random Forest Regressor and pipeline preprocessing.

3. **Restaurant Recommendation System**
   - Recommends top 5 similar restaurants based on user preferences.
   - Uses TF-IDF vectorization and cosine similarity for text-based recommendations.

4. **Interactive Map Visualization**
   - Displays a dynamic, clustered HTML map of restaurant distribution.
   - Helps users explore restaurant locations visually.

---

## Project Structure

Restaurant-Intelligence-Platform/
│
├── app.py # Main Streamlit application
├── requirements.txt # List of dependencies
├── README.md
│
├── models/ # Trained model files
│ ├── cuisine_model.pkl
│ ├── rating_model.pkl
│ ├── vectorizer.pkl
│ ├── label_encoder.pkl
│ └── dataframe.pkl
│
├── Tasks/ # All Jupyter notebooks + map
│ ├── task-3&4.ipynb # Cuisine classification notebook
│ ├── AverageRatingPrediction.ipynb # Rating prediction notebook
│ ├── Recommendation_System.ipynb # Recommender system notebook
│ └── Restaurant_Distribution_Clustered_Full.html # Interactive map


---

## ML Models Used

| Task                     | Algorithm Used             | Output `.pkl` File                |
|--------------------------|----------------------------|-----------------------------------|
| Cuisine Classification   | GradientBoostingClassifier | `cuisine_model.pkl`               |
| Rating Prediction        | RandomForestRegressor      | `rating_model.pkl`                |
| Recommendation System    | TF-IDF + Cosine Similarity | `dataframe.pkl` |

---

## Deployment Instructions

This app is deployed using **Streamlit Community Cloud**.

### Steps to Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Restaurant-Intelligence-Platform.git
   cd Restaurant-Intelligence-Platform

   Install dependencies:

2. Install Dependencies
   pip install -r requirements.txt

3. Run the app
   streamlit run app.py

<img width="1789" height="833" alt="image" src="https://github.com/user-attachments/assets/398342d4-099c-406b-9ad9-465b631f64cb" />
<img width="1737" height="823" alt="image" src="https://github.com/user-attachments/assets/08298f98-00ed-4852-8071-2e59bee21b96" />
<img width="1761" height="787" alt="image" src="https://github.com/user-attachments/assets/ff321c18-d8cf-4fa3-a11b-45a19e9799e6" />
<img width="1817" height="841" alt="image" src="https://github.com/user-attachments/assets/165aeee9-cc33-4ff1-bbb4-27e7ac6151ea" />



# Tech Stack

Frontend: Streamlit

Backend: Python, Scikit-learn, Pandas, Numpy

Visualization: Matplotlib, Seaborn, Folium

ML Models: Gradient Boosting, Random Forest, TF-IDF + Cosine Similarity

Deployment: Streamlit Cloud

# Author
Bhavay Khandelwal
B.Tech Final Year | Aspiring Data Scientist
Email:-bl1183757@gmail.com
LinkedIn:-https://www.linkedin.com/in/bhavay-khandelwal-11a9b628a/
GitHub:-https://www.github.com/BL1183757

# Show Your Support
If you like this project:

Star this repository

Fork it for later reference

Share it with your peers
  
# Note
This is a portfolio project created for learning and demonstration purposes. Accuracy may vary due to limited dataset quality or scope.


