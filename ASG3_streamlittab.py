import streamlit as st
import pandas as pd
import numpy as np
import requests
import snowflake.connector
from urllib.error import URLError

st.set_page_config(page_title='INVEMP Tasty Bytes Group 5', page_icon='🍖🍕🍜')

st.sidebar.title("INVEMP: Inventory/Warehouse Management & Prediction on Sales per Menu Item")
st.sidebar.markdown("This web app allows you to explore the internal inventory of Tasty Bytes. You can explore these functions in the web app (Description of Page)")

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Prediction A', 'Prediction B', 'Prediction C', 'Prediction D', 'Prediction E'])

with tab1:
  #Tab 1 code here
  #Hector/Shahid
  st.write("Hello")
with tab2:
  #Tab 2 code here
  #Hector/Shahid
  st.write("Hello")
with tab3:
  #Tab 3 code here
  st.write("Hello")

with tab4:
      import streamlit as st
      import pandas as pd
      import numpy as np
      import pydeck as pdk
      import joblib
      from joblib import load
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
      
      
      data = load_data()
      github_url = "https://github.com/kaylaong883/ASG3_streamlit_tab/raw/main/y2022_data_withtid.zip"
      maintable = read_csv_from_zipped_github(github_url)

      with open('xgbr_gs.pkl', 'rb') as file:
        xgbr_gs = joblib.load(file)
        
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
      
      season_mapping = {'WINTER': 0, 'SPRING': 1, 'SUMMER': 2, 'AUTUMN': 3}
      season_reverse_mapping = {v: k for k, v in season_mapping.items()}
      season_labels = list(season_mapping.keys())
      season_values = list(season_mapping.values())
      
      city_mapping = {'San Mateo': 0, 'Denver': 1, 'Seattle': 2, 'New York City': 3, 'Boston': 4}
      city_reverse_mapping = {v: k for k, v in city_mapping.items()}
      city_labels = list(city_mapping.keys())
      city_values = list(city_mapping.values())
      
      itemcat_mapping = {'Dessert': 0, 'Beverage': 1, 'Main': 2, 'Snack': 3}
      itemcat_reverse_mapping = {v: k for k, v in itemcat_mapping.items()}
      itemcat_labels = list(itemcat_mapping.keys())
      
      menut_mapping = {'Ice Cream': 0, 'Grilled Cheese': 1, 'BBQ': 2, 'Tacos': 3, 'Chinese': 4, 'Poutine': 5, 'Hot Dogs': 6, 'Vegetarian': 7, 'Crepes': 8, 'Sandwiches': 9, 'Ramen': 10, 'Ethiopian': 11, 'Gyros': 12, 'Indian': 13, 'Mac & Cheese': 14}
      menut_reverse_mapping = {v: k for k, v in menut_mapping.items()}
      menut_labels = list(menut_mapping.keys())
      
      truckb_mapping = {'Freezing Point': 0, 'The Mega Melt': 1, 'Smoky BBQ': 2, "Guac n' Roll": 3, 'Peking Truck': 4, 'Revenge of the Curds': 5, 'Not the Wurst Hot Dogs': 6, 'Plant Palace': 7, 'Le Coin des Crêpes': 8, 'Better Off Bread': 9, 'Kitakata Ramen Bar': 10, 'Tasty Tibs': 11, 'Cheeky Greek': 12, "Nani's Kitchen": 13, 'The Mac Shack': 14}
      truckb_reverse_mapping = {v: k for k, v in truckb_mapping.items()}
      truckb_labels = list(truckb_mapping.keys())
      truckb_values = list(truckb_mapping.values())
      
      menuitem_mapping = {'Mango Sticky Rice': 0, 'Popsicle': 1, 'Waffle Cone': 2, 'Sugar Cone': 3, 'Two Scoop Bowl': 4, 'Lemonade': 5, 'Bottled Water': 6, 'Ice Tea': 7, 'Bottled Soda': 8, 'Ice Cream Sandwich': 9, 'The Ranch': 10, 'Miss Piggie': 11, 
                          'The Original': 12, 'Three Meat Plate': 13, 'Fried Pickles': 14, 'Two Meat Plate': 15, 'Spring Mix Salad': 16, 'Rack of Pork Ribs': 17, 'Pulled Pork Sandwich': 18, 'Fish Burrito': 19, 'Veggie Taco Bowl': 20, 'Chicken Burrito': 21, 'Three Taco Combo Plate': 22,
                          'Two Taco Combo Plate': 23, 'Lean Burrito Bowl': 24, 'Combo Lo Mein': 25, 'Wonton Soup': 26, 'Combo Fried Rice': 27, 'The Classic': 28, 'The Kitchen Sink': 29, 'Mothers Favorite': 30, 'New York Dog': 31, 'Chicago Dog': 32, 'Coney Dog': 33, 'Veggie Burger': 34,
                          'Seitan Buffalo Wings': 35, 'The Salad of All Salads': 36, 'Breakfast Crepe': 37, 'Chicken Pot Pie Crepe': 38, 'Crepe Suzette': 39, 'Hot Ham & Cheese': 40, 'Pastrami': 41, 'Italian': 42, 'Creamy Chicken Ramen': 43, 'Spicy Miso Vegetable Ramen': 44, 'Tonkotsu Ramen': 45,
                          'Veggie Combo': 46, 'Lean Beef Tibs': 47, 'Lean Chicken Tibs': 48, 'Gyro Plate': 49, 'Greek Salad': 50, 'The King Combo': 51, 'Tandoori Mixed Grill': 52, 'Lean Chicken Tikka Masala': 53, 'Combination Curry': 54, 'Lobster Mac & Cheese': 55, 'Standard Mac & Cheese': 56, 
                          'Buffalo Mac & Cheese': 57}
      
      menuitem_reverse_mapping = {v: k for k, v in menuitem_mapping.items()}
      menuitem_labels = list(menuitem_mapping.keys())
  
      def get_season():
        season = st.selectbox('Select a season', season_labels)
        return season

      season_input = get_season()
      season_int = season_mapping[season_input]
      
      def get_truckb():
          truckb = st.selectbox('Select a Truck Brand Name', truckb_labels)
          return truckb
          
      truckb_input = get_truckb()
      truckb_int = truckb_mapping[truckb_input]
  
      filter_rows = []
      for index, row in maintable.iterrows():
        if (season_input in row['SEASON']) & (truckb_input in row['TRUCK_BRAND_NAME']):
          filter_rows.append(row)
          
      filter_df = pd.DataFrame(filter_rows, columns=maintable.columns)

      # get unique values of trucks for filtered data
      truck_array = filter_df['TRUCK_ID'].unique()
      
      # user input for number of trucks
      user_truck_input = st.number_input("Enter the number of trucks you want to implement", min_value=0, max_value=len(truck_array))
      st.write("No. of trucks:", user_truck_input)

      if user_truck_input > len(truck_array):
          st.error("Error: Number of trucks exceeds available options")

      user_choose_truck = truck_array[:user_truck_input]

      # filter data again to only contain the number of trucks the user choose
      filter_user = filter_df[filter_df['TRUCK_ID'].isin(user_choose_truck)]

      st.write("Truck numbers:", user_choose_truck)
      if st.button('Predict Sales'):
        filter_user['VALUE'] = 0
        filter_user['discount_10%'] = 0
        truck_list = filter_user['TRUCK_ID']
        prediction_table = filter_user.drop(columns=['TOTAL_SALES_PER_ITEM','TRUCK_ID'])

        # Change values to numeric for model to predict
        ## map values to put in dataframe
        prediction_table['SEASON'] = prediction_table['SEASON'].map(season_mapping)
        prediction_table['CITY'] = prediction_table['CITY'].map(city_mapping)
        prediction_table['ITEM_CATEGORY'] = prediction_table['ITEM_CATEGORY'].map(itemcat_mapping)
        prediction_table['MENU_TYPE'] = prediction_table['MENU_TYPE'].map(menut_mapping)
        prediction_table['TRUCK_BRAND_NAME'] = prediction_table['TRUCK_BRAND_NAME'].map(truckb_mapping)
        prediction_table['MENU_ITEM_NAME'] = prediction_table['MENU_ITEM_NAME'].map(menuitem_mapping)
        column_names = []
        column_names = prediction_table.columns.tolist()

        input_data = column_names
        input_df = prediction_table
        prediction = xgbr_gs.predict(input_df)
        output_data = pd.DataFrame(input_df, columns = input_df.columns)
        output_data['PREDICTED_PRICE'] = prediction 
        
        output_data = pd.concat([truck_list, output_data], axis=1)

        output_data['SEASON'] = output_data['SEASON'].map(season_reverse_mapping)
        output_data['CITY'] = output_data['CITY'].map(city_reverse_mapping)
        output_data['ITEM_CATEGORY'] = output_data['ITEM_CATEGORY'].map(itemcat_reverse_mapping)
        output_data['MENU_TYPE'] = output_data['MENU_TYPE'].map(menut_reverse_mapping)
        output_data['TRUCK_BRAND_NAME'] = output_data['TRUCK_BRAND_NAME'].map(truckb_reverse_mapping)
        output_data['MENU_ITEM_NAME'] = output_data['MENU_ITEM_NAME'].map(menuitem_reverse_mapping)
        st.write(output_data)
  
      # def get_CITY():
      # city = st.selectbox('Select a City', city_labels)
      # return city
          
      # city_input = get_CITY()
      # city_int = city_mapping[city_input]
          
    


with tab5:
  #Tab 5 code here
  st.write("Hello")



