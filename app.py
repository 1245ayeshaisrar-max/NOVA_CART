import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Shopping Assistant", page_icon="🛒", layout="wide")

# --- 1. DATA LOADING & PREPARATION ---
@st.cache_data # Ensures data loads once and stays fast
def load_and_prepare_data():
    try:
        # Load the file (Ensure 'product_ratings.csv' is in your GitHub repo)
        df = pd.read_csv('product_ratings.csv')
        
        # Remove empty columns
        df = df.dropna(axis=1, how='all')

        # Standardize column names
        df.columns = [c.strip() for c in df.columns]
        df.rename(columns={
            'Product Name': 'name',
            'CATEGORY': 'category',
            'USAGE': 'usage',
            'TARGET USER': 'target',
            'PRICE RANGE': 'price'
        }, inplace=True)

        # Filter out repeated headers and fill missing values
        df = df[df['name'].str.lower() != 'product name'].reset_index(drop=True)
        df = df.fillna('N/A')

        # Generate Metadata for TF-IDF
        df['metadata'] = df['category'] + " " + df['usage'] + " " + df['target']
        
        return df
    except FileNotFoundError:
        st.error("CSV file not found. Please ensure 'product_ratings.csv' is in your repository.")
        return None

df = load_and_prepare_data()

# --- 2. RECOMMENDATION ENGINE ---
@st.cache_resource # Keeps the similarity matrix in memory
def build_engine(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['metadata'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

if df is not None:
    cosine_sim = build_engine(df)

def get_recommendations(product_query):
    try:
        product_query = product_query.strip().lower()
        matches = df[df['name'].str.lower() == product_query]
        
        if matches.empty:
            return None, None
            
        # Get the searched item details
        searched_item = matches.iloc[[0]]
        idx = matches.index[0]
        
        # Get top 5 similarities
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        
        recommendations = df.iloc[[i[0] for i in sim_scores]]
        return searched_item, recommendations
    except Exception:
        return None, None

# --- 3. USER INTERFACE ---
st.title("🛒 AI Shopping Assistant")
st.write("Find a product and see what other users are buying.")

# Search Input
user_query = st.text_input("Enter Product Name:", placeholder="e.g., serum, facewash, calculator...")

if user_query:
    target, recs = get_recommendations(user_query)
    
    if target is not None:
        # Define columns to display
        display_cols = ['name', 'category', 'usage', 'target', 'price']
        
        # --- SECTION 1: SEARCHED ITEM ---
        st.markdown("### 📍 YOUR SEARCHED ITEM")
        # Display as a clean table
        st.table(target[display_cols].rename(columns=lambda x: x.upper()))

        st.markdown("---") # Visual separator

        # --- SECTION 2: RECOMMENDATIONS ---
        st.markdown("### 🌟 OTHER USERS ALSO LIKED")
        # Display recommendations
        st.table(recs[display_cols].rename(columns=lambda x: x.upper()))
        
    else:
        st.warning(f"Product '{user_query}' was not found. Please try another search.")

# Sidebar Info
st.sidebar.header("How it works")
st.sidebar.write("This system uses **Collaborative Logic** (TF-IDF + Cosine Similarity) to recommend items based on their Category, Usage, and Target User profile.")
