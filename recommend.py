import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import streamlit as st

df = pd.read_csv('products.csv')

required_columns = ['Product ID', 'Product Name', 'Category', 'Price (NPR)']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns in 'products.csv': {missing_columns}")

num_users = 5
ratings_data = [
    [user_id, product_id, np.random.randint(1, 6)]
    for user_id in range(1, num_users + 1)
    for product_id in df['Product ID']
]
ratings_df = pd.DataFrame(ratings_data, columns=['user_id', 'product_id', 'rating'])

user_item_matrix = ratings_df.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)

user_knn = NearestNeighbors(metric='cosine', algorithm='brute').fit(user_item_matrix)
item_knn = NearestNeighbors(metric='cosine', algorithm='brute').fit(user_item_matrix.T) 

# Recommendation function
def recommend_products(user_id, top_n=5, method='user'):
    if user_id not in ratings_df['user_id'].unique():
        raise ValueError(f"User ID {user_id} not found in the dataset.")
    
    if method == 'user':
        distances, indices = user_knn.kneighbors([user_item_matrix.loc[user_id]], n_neighbors=min(top_n + 1, len(user_item_matrix)))
        similar_users = indices.flatten()[1:]
        recommendations = user_item_matrix.iloc[similar_users].mean(axis=0).sort_values(ascending=False)
    else:
        user_ratings = user_item_matrix.loc[user_id].values
        selected_ratings = user_ratings[:5]  
        selected_ratings = selected_ratings.reshape(1, -1) 
        
        distances, indices = item_knn.kneighbors(selected_ratings, n_neighbors=min(top_n, len(user_item_matrix.columns)))
        similar_items = indices.flatten()
        recommendations = user_item_matrix.iloc[:, similar_items].mean(axis=1).sort_values(ascending=False)

    return [
        {
            'Product Name': df.loc[df['Product ID'] == product_id, 'Product Name'].values[0],
            'Category': df.loc[df['Product ID'] == product_id, 'Category'].values[0],
            'Price (NPR)': df.loc[df['Product ID'] == product_id, 'Price (NPR)'].values[0],
            'Predicted Rating': recommendations[product_id]
        }
        for product_id in recommendations.index[:top_n]
    ]

# Streamlit UI
st.title("Collaborative Filtering Product Recommender")

st.sidebar.header("User Settings")
user_id = st.sidebar.selectbox("Select User ID", ratings_df['user_id'].unique())
method = st.sidebar.radio("Recommendation Method", ('user', 'item'))
top_n = st.sidebar.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

if st.sidebar.button("Generate Recommendations"):
    try:
        recommendations = recommend_products(user_id=user_id, top_n=top_n, method=method)
        st.subheader(f"Top {top_n} Recommended Products for User {user_id} ({'User-based' if method == 'user' else 'Item-based'}):")
        for rec in recommendations:
            st.write(f"**Product Name:** {rec['Product Name']}  \n"
                     f"**Category:** {rec['Category']}  \n"
                     f"**Price (NPR):** {rec['Price (NPR)']}  \n"
                     f"**Predicted Rating:** {rec['Predicted Rating']:.2f}")
    except ValueError as e:
        st.error(e)
