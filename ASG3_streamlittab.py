import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
#import joblib
#from joblib import load
import pickle
from xgboost import XGBRegressor 
import requests
import zipfile
import io

@st.cache_data
def load_data():
    # First load the original airbnb listtings dataset
    data = pd.read_csv("final_data_noscaler.csv") #use this for the original dataset, before transformations and cleaning
    return data


def read_csv_from_zipped_github(url):
    # Send a GET request to the GitHub URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Create a BytesIO object from the response content
        zip_file = io.BytesIO(response.content)

        # Extract the contents of the zip file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Assume there is only one CSV file in the zip archive (you can modify this if needed)
            csv_file_name = zip_ref.namelist()[0]
            with zip_ref.open(csv_file_name) as csv_file:
                # Read the CSV data into a Pandas DataFrame
                df = pd.read_csv(csv_file)

        return df
    else:
        st.error(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
        return None

def main():
    st.title("Read CSV from Zipped File on GitHub")

    # Replace the 'github_url' variable with the actual URL of the zipped CSV file on GitHub
    github_url = "https://github.com/kaylaong883/ASG3_streamlit_tab/blob/main/snowflake_data.zip"
    df = read_csv_from_zipped_github(github_url)


data = load_data()
maintable = main()

# Define the app title and favicon
st.title('Seasonal Menu Variations') 
st.subheader('Predict')
st.markdown("This tab allows you to make predictions on the sales of the trucks  based on the neighbourhood and room type. The model used is a Random Forest Regressor trained on the Airbnb Singapore listings dataset.")
st.write('Choose a neighborhood group, neighborhood, and room type to get the predicted average price.')

st.header('Truck Menu Available')

data = {
    'Truck Name': ["Guac n' Roll", "Tasty Tibs", "The Mac Shack", "Peking Truck", "Le Coin des Crêpes", "Freezing Point", "Nani's Kitchen", "The Mega Melt", "Better Off Bread", "Not the Wurst Hot Dogs", "Plant Palace", "Cheeky Greek", "Revenge of the Curds", "Kitakata Ramen Bar", "Smoky BBQ"],
    'Menu Name': ['Tacos', 'Ethiopian', 'Mac & Cheese', 'Chinese', 'Crepes', 'Ice Cream', 'Indian', 'Grilled Cheese', 'Sandwiches', 'Hot Dogs', 'Vegetarian', 'Gyros', 'Poutine', 'Ramen', 'BBQ']
}
truck_menu_table = pd.DataFrame(data)

# Display the DataFrame as a table
st.table(truck_menu_table)

# Define the user input functions

tbn_list = []
for tbn in data['TRUCK_BRAND_NAME']:
    if tbn not in tbn_list:
        tbn_list.append(tbn)
    else:
        continue 

number = [] 
for i in range(len(tbn_list)):
    number.append(i)

tbn_dict = dict(zip(tbn_list, number))
tbn_dict_reverse_mapping = {v: k for k, v in tbn_dict.items()}
tbn_labels = list(tbn_dict.keys())
tbn_labels = [tbn_dict_reverse_mapping[i] for i in sorted(tbn_dict_reverse_mapping.keys())]

def get_tbn_group():
    tbn_group = st.selectbox('Select a truck', tbn_labels)
    return tbn_group

# Define the user input fields
ng_input = get_neighbourhood_group()

# don't run anything past here while we troubleshoot
# streamlit.stop()
