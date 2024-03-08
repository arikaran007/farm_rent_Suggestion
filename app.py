import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle

farm = pd.read_csv('farm.csv')

farm_prod = farm[['applicant_id', 'applicant_district', 'product', 'Distance_to_Rental_Locations']]
farm_prod.drop_duplicates(['applicant_id', 'product'], inplace=True)
farm_pivot = farm_prod.pivot_table(columns='applicant_id', index='product', values='Distance_to_Rental_Locations')
farm_pivot.fillna(0, inplace=True)
farm_sparse = csr_matrix(farm_pivot)

with open('farm_mod.pkl', 'rb') as f:
    model = pickle.load(f)

def recommend_farm(product):
    try:
        product_index = np.where(farm_pivot.index == product)[0][0]
        _, suggestion = model.kneighbors(farm_pivot.iloc[product_index, :].values.reshape(1, -1), n_neighbors=6)

        st.write(f"You searched '{product}'")
        st.write("The suggested products are:")
        for product_index in suggestion[0]:
            suggested_product = farm_pivot.index[product_index]
            if suggested_product != product:
                st.write(suggested_product)
    except IndexError:
        st.error("Product not found!")

st.title("Farm Machinery Rental System")

product_input = st.text_input("Enter the product:")
if st.button("Recommend"):
    if product_input:
        recommend_farm(product_input)
    else:
        st.warning("Please enter a product!")
