import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
# import joblib
# from joblib import load
import pickle
from sklearn import preprocessing

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

st.title('Seasonal Menu Variations')

st.header('Truck Menu Available')

data = {
    'Truck Name': ["Guac n' Roll", "Tasty Tibs", "The Mac Shack", "Peking Truck", "Le Coin des Crêpes", "Freezing Point", "Nani's Kitchen", "The Mega Melt", "Better Off Bread", "Not the Wurst Hot Dogs", "Plant Palace", "Cheeky Greek", "Revenge of the Curds", "Kitakata Ramen Bar", "Smoky BBQ"],
    'Menu Name': ['Tacos', 'Ethiopian', 'Mac & Cheese', 'Chinese', 'Crepes', 'Ice Cream', 'Indian', 'Grilled Cheese', 'Sandwiches', 'Hot Dogs', 'Vegetarian', 'Gyros', 'Poutine', 'Ramen', 'BBQ']
}
truck_menu_table = pd.DataFrame(data)

# Display the DataFrame as a table
st.table(truck_menu_table)

# read csv
# Provide the path to the CSV file
# csv_file_path = "final_data_noscaler.csv"

# Read the CSV file into a DataFrame using Pandas
# data = pandas.read_csv(csv_file_path)

# Display the DataFrame as a table in the Streamlit app
st.write("CSV File Contents:")
st.dataframe(data)

zip_file_url = "https://github.com/kaylaong883/ASG3_streamlit_tab/raw/main/snowflake_data.zip"

response = requests.get(zip_file_url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))

# Assuming the zip file contains a single CSV file
csv_file_name = zip_file.namelist()[0]
with zip_file.open(csv_file_name) as file:
    df = pandas.read_csv(file)

# Display the DataFrame as a table in the Streamlit app
streamlit.write("Zip File CSV File Contents:")
streamlit.dataframe(df)

streamlit.header('🍌🥭 Build Your Own Fruit Smoothie 🥝🍇')

# import pandas
my_fruit_list = pandas.read_csv("https://uni-lab-files.s3.us-west-2.amazonaws.com/dabw/fruit_macros.txt")
my_fruit_list = my_fruit_list.set_index('Fruit')

# Let's put a pick list here so they can pick the fruit they want to include 
fruits_selected = streamlit.multiselect("Pick some fruits:", list(my_fruit_list.index),['Avocado','Strawberries'])
fruits_to_show = my_fruit_list.loc[fruits_selected]

# Display the table on the page.
streamlit.dataframe(fruits_to_show)

#create the repeatable code block (called a function)
def get_fruityvice_data(this_fruit_choice):
  fruityvice_response = requests.get("https://fruityvice.com/api/fruit/" + this_fruit_choice)
  fruityvice_normalized = pandas.json_normalize(fruityvice_response.json())
  return (fruityvice_normalized)

#New Section to display fruityvice api response
streamlit.header("Fruityvice Fruit Advice!")
try:
  fruit_choice = streamlit.text_input('What fruit would you like information about?','Kiwi')
  if not fruit_choice:
    streamlit.error("Please select a fruit to get information.")
  else:
    back_from_function = get_fruityvice_data(fruit_choice)
    streamlit.dataframe(back_from_function)

except URLError as e:
  streamlit.error()   

# import snowflake.connector
streamlit.header("View Our Fruit List - Add Your Favorites!")
#Snowflake-related functions
def get_fruit_load_list():
  with my_cnx.cursor() as my_cur:
    my_cur.execute("select * from fruit_load_list")
    return my_cur.fetchall()
  
# Add a button to load the fruit
if streamlit.button('Get Fruit List'):
  my_cnx = snowflake.connector.connect(**streamlit.secrets["snowflake"])
  my_data_rows = get_fruit_load_list()
  my_cnx.close()
  streamlit.dataframe(my_data_rows)

# Allow the end user to add a fruit to the list
def insert_row_snowflake(new_fruit):
  with my_cnx.cursor() as my_cur:
    my_cur.execute("insert into fruit_load_list values ('" + new_fruit +"')")
    return "Thanks for adding " + new_fruit
 
add_my_fruit = streamlit.text_input('What fruit would you like to add?','jackfruit')
if streamlit.button('Add a Fruit to the List'):
  my_cnx = snowflake.connector.connect(**streamlit.secrets["snowflake"])
  back_from_function = insert_row_snowflake(add_my_fruit)
  streamlit.text(back_from_function)

# don't run anything past here while we troubleshoot
# streamlit.stop()
