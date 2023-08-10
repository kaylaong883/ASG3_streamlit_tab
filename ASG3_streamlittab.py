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
      
      github_url_WO2022 = "https://github.com/kaylaong883/ASG3_streamlit_tab/raw/main/woy2022_data.zip"
      df_WO2022 = read_csv_from_zipped_github(github_url_WO2022)
      github_url = "https://github.com/kaylaong883/ASG3_streamlit_tab/raw/main/y2022_qty_data.zip"
      maintable = read_csv_from_zipped_github(github_url)

      with open('xgbr_gs.pkl', 'rb') as file:
        xgbr_gs = joblib.load(file)
        
      # Define the app title and favicon
      st.title('🚚🍝 Implementation of Additional Trucks 🚛🥗') 
      st.subheader('Predict')
      st.markdown("This interactive tab serves as a strategic tool for evaluating the potential benefits of introducing additional trucks during specific seasons. It empowers you to make informed predictions regarding total truck sales based on varying fleet sizes. By inputting the desired number of trucks for implementation, you can project the expected total sales. Furthermore, this tool calculates the average sales for each truck, facilitating a meaningful comparison with historical sales data from previous years. This insightful analysis aids in determining the viability and profitability of expanding your fleet during specific seasons, enabling you to make well-informed decisions for your business growth strategy. ")
      
      st.header('Truck Menu Available')
      
      data = {
          'Truck Name': ["Guac n' Roll", "Tasty Tibs", "The Mac Shack", "Peking Truck", "Le Coin des Crêpes", "Freezing Point", "Nani's Kitchen", "The Mega Melt", "Better Off Bread", "Not the Wurst Hot Dogs", "Plant Palace", "Cheeky Greek", "Revenge of the Curds", "Kitakata Ramen Bar", "Smoky BBQ"],
          'Menu Name': ['Tacos', 'Ethiopian', 'Mac & Cheese', 'Chinese', 'Crepes', 'Ice Cream', 'Indian', 'Grilled Cheese', 'Sandwiches', 'Hot Dogs', 'Vegetarian', 'Gyros', 'Poutine', 'Ramen', 'BBQ']
      }
      truck_menu_table = pd.DataFrame(data)
      
      # Display the DataFrame as a table
      st.table(truck_menu_table)

      st.write("Select a season, the brand name of the truck you're interested in, and specify the desired number of trucks for implementation. This will provide you with the projected average sales per truck following the implementation.")
            
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

      filter_df = filter_df.drop(columns=['TOTAL_SALES_PER_ITEM','DATE'])

      # user input for number of trucks
      user_truck_input = st.number_input("Enter the number of trucks you want to implement", min_value=0, max_value=100)
      st.write("No. of trucks:", user_truck_input)

      # GENERATE RECORD FOR NEW TRUCKS
      # Initialize an empty list to store generated data
      data = []
      
      # List of possible options
      location = filter_df['LOCATION_ID'].unique()
      
      min_quantity = filter_df['TOTAL_QTY_SOLD'].min()
      max_quantity = filter_df['TOTAL_QTY_SOLD'].max()
      
      shift_no = filter_df['SHIFT_NUMBER'].unique()
      
      city = filter_df['CITY'].unique()
      
      subcat = filter_df['SUBCATEGORY'].unique()
      
      menu_type = filter_df['MENU_TYPE'].unique()
      
      min_air = filter_df['AVG_TEMPERATURE_AIR_2M_F'].min()
      max_air = filter_df['AVG_TEMPERATURE_AIR_2M_F'].max()
      
      min_wb = filter_df['AVG_TEMPERATURE_WETBULB_2M_F'].min()
      max_wb = filter_df['AVG_TEMPERATURE_WETBULB_2M_F'].max()
      
      min_dp = filter_df['AVG_TEMPERATURE_DEWPOINT_2M_F'].min()
      max_dp = filter_df['AVG_TEMPERATURE_DEWPOINT_2M_F'].max()
      
      min_wc = filter_df['AVG_TEMPERATURE_WINDCHILL_2M_F'].min()
      max_wc = filter_df['AVG_TEMPERATURE_WINDCHILL_2M_F'].max()
      
      min_ws = filter_df['AVG_WIND_SPEED_100M_MPH'].min()
      max_ws = filter_df['AVG_WIND_SPEED_100M_MPH'].max()
      
      
      # Initialize an empty dictionary to store item details
      item_details = {}
      
      for index, row in filter_df.iterrows():
          menu_item_name = row['MENU_ITEM_NAME']
          item_category = row['ITEM_CATEGORY']
          cost_of_goods = row['COG_PER_ITEM_USD']
          item_price = row['ITEM_PRICE']
      
          item_details[menu_item_name] = {
              'ITEM_CATEGORY': item_category,
              'COG_PER_ITEM_USD': cost_of_goods,
              'ITEM_PRICE': item_price
          }
          
      
      for user in range(user_truck_input):
          TRUCK_ID = user + 101  # Starting truck ID for each user
      
          # Generate 700 rows of data
          for i in range(700):
      
              LOCATION_ID = np.random.choice(location)
      
              TOTAL_QTY_SOLD = np.random.randint(min_quantity, max_quantity + 1)
              
              SHIFT_NUMBER = np.random.choice(shift_no)
              
              CITY = np.random.choice(city)
              
              SUBCATEGORY = np.random.choice(subcat)
              
              MENU_TYPE = np.random.choice(menu_type)
              
              TRUCK_BRAND_NAME = truckb_input
              
              AVG_TEMPERATURE_AIR_2M_F = np.random.randint(min_air, max_air + 1)
              
              AVG_TEMPERATURE_WETBULB_2M_F = np.random.randint(min_wb , max_wb + 1)
              
              AVG_TEMPERATURE_DEWPOINT_2M_F = np.random.randint(min_dp, max_dp + 1)
              
              AVG_TEMPERATURE_WINDCHILL_2M_F = np.random.randint(min_wc, max_wc + 1)
              
              AVG_WIND_SPEED_100M_MPH = np.random.randint(min_ws, max_ws + 1)
              
              SEASON = season_input
      
              MENU_ITEM_NAME = np.random.choice(filter_df['MENU_ITEM_NAME'])
              ITEM_CATEGORY = item_details[MENU_ITEM_NAME]['ITEM_CATEGORY']
              COG_PER_ITEM_USD = item_details[MENU_ITEM_NAME]['COG_PER_ITEM_USD']
              ITEM_PRICE = item_details[MENU_ITEM_NAME]['ITEM_PRICE']
              
              VALUE = 0
              
              DISCOUNT = ITEM_PRICE
      
              data.append({
                  'LOCATION_ID':LOCATION_ID,
                  'TRUCK_ID':TRUCK_ID,
                  'TOTAL_QTY_SOLD':TOTAL_QTY_SOLD,
                  'SHIFT_NUMBER':SHIFT_NUMBER,
                  'CITY':CITY,
                  'ITEM_CATEGORY': ITEM_CATEGORY,
                  'SUBCATEGORY':SUBCATEGORY,
                  'MENU_TYPE':MENU_TYPE,
                  'TRUCK_BRAND_NAME':TRUCK_BRAND_NAME,
                  'MENU_ITEM_NAME': MENU_ITEM_NAME,
                  'AVG_TEMPERATURE_AIR_2M_F':AVG_TEMPERATURE_AIR_2M_F,
                  'AVG_TEMPERATURE_WETBULB_2M_F':AVG_TEMPERATURE_WETBULB_2M_F,
                  'AVG_TEMPERATURE_DEWPOINT_2M_F':AVG_TEMPERATURE_DEWPOINT_2M_F,
                  'AVG_TEMPERATURE_WINDCHILL_2M_F':AVG_TEMPERATURE_WINDCHILL_2M_F,
                  'AVG_WIND_SPEED_100M_MPH':AVG_WIND_SPEED_100M_MPH,
                  'SEASON':SEASON,
                  'COG_PER_ITEM_USD':COG_PER_ITEM_USD,
                  'ITEM_PRICE': ITEM_PRICE,
                  'VALUE':VALUE,
                  'discount_10%':DISCOUNT
              })
      
      # Create a DataFrame from the generated data
      df_generated = pd.DataFrame(data)

      # JOIN filter_df and df_generated
      frames = [filter_df, df_generated]
      prediction_table = pd.concat(frames)

      if st.button('Predict Sales'):
        prediction_table['VALUE'] = 0
        prediction_table['discount_10%'] = prediction_table['ITEM_PRICE']
        truck_list = prediction_table['TRUCK_ID'] 
        qty_list = prediction_table['TOTAL_QTY_SOLD']
        prediction_table = prediction_table.drop(columns=['TRUCK_ID','TOTAL_QTY_SOLD'])

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
        
        output_data = pd.concat([truck_list, qty_list, output_data], axis=1)

        output_data['SEASON'] = output_data['SEASON'].map(season_reverse_mapping)
        output_data['CITY'] = output_data['CITY'].map(city_reverse_mapping)
        output_data['ITEM_CATEGORY'] = output_data['ITEM_CATEGORY'].map(itemcat_reverse_mapping)
        output_data['MENU_TYPE'] = output_data['MENU_TYPE'].map(menut_reverse_mapping)
        output_data['TRUCK_BRAND_NAME'] = output_data['TRUCK_BRAND_NAME'].map(truckb_reverse_mapping)
        output_data['MENU_ITEM_NAME'] = output_data['MENU_ITEM_NAME'].map(menuitem_reverse_mapping)

        # filter output data to only contain predicted trucks
        output_data = output_data[output_data['TRUCK_ID'] > 100]
        st.write(output_data)

        # 2022 DATA
        filter_rows_2022 = []
        for index, row in maintable.iterrows():
          if (season_input in row['SEASON']) & (truckb_input in row['TRUCK_BRAND_NAME']):
            filter_rows_2022.append(row)
          
        filter_rows_2022 = pd.DataFrame(filter_rows_2022, columns=maintable.columns)
      
        total_sales_of_trucks_2022 = 0
        truck_avail_2022 = filter_rows_2022['TRUCK_ID'].unique()
        
        # Create a list to hold truck information
        truck_info_2022 = []
        
        for truck in truck_avail_2022:
            total_sales_2022 = filter_rows_2022[filter_rows_2022['TRUCK_ID'] == truck]['TOTAL_SALES_PER_ITEM'].sum()
            truck_info_2022.append({'Truck': truck, 'Total Sales': total_sales_2022})

        truck_info_2022_display = truck_info_2022.copy()
        
        # truck sales with predicted truck
        total_sales_of_trucks = 0
        trucks_available = output_data['TRUCK_ID'].unique()

        # Create a list to hold truck information
        truck_info = []

        for truck in trucks_available:
            total_sales = output_data[output_data['TRUCK_ID'] == truck]['PREDICTED_PRICE'].sum()
            truck_info.append({'Truck': truck, 'Total Sales': total_sales})

        truck_info_2022.extend(truck_info)
        truck_info = truck_info_2022

        # Display truck information in a table
        #st.table(pd.DataFrame(truck_info))
        truck_info = pd.DataFrame(truck_info)
        
        # Filter rows where truck_id > 100
        highlighted_rows = truck_info[truck_info['truck_id'] > 100]
        
        # Display the table with highlighted rows
        st.table(highlighted_rows.style.apply(lambda x: ['background: yellow' if val > 100 else '' for val in x], axis=1))
        
        # Calculate total and average sales
        total_sales_of_trucks = sum(info['Total Sales'] for info in truck_info)
        average_sales = total_sales_of_trucks / (len(trucks_available) + 5 )
        
        # Print total sales for all trucks combined
        st.subheader(f"Total sales for all {len(trucks_available) + 5} trucks: ${total_sales_of_trucks:.2f}")
        
        # Display average sales
        st.subheader(f"Average sales for each truck: ${average_sales:.2f}")


        # FOR COMPARISON WITH 2022 DATA
        st.header(f"Comparison with 2022 data")
        # st.write(filter_rows_2022)
        
        # PRINTING OF 2022 TRUCK INFO
        # Calculate total and average sales
        total_sales_of_trucks_2022 = sum(info['Total Sales'] for info in truck_info_2022_display)
        average_sales_2022 = total_sales_of_trucks_2022 / len(truck_avail_2022)
      
        # Display truck information in a table
        st.table(pd.DataFrame(truck_info_2022_display))
        
        # Print total sales for all trucks combined
        st.subheader(f"Total sales for all {len(truck_avail_2022)} trucks: ${total_sales_of_trucks_2022:.2f}")
        
        # Display average sales
        st.subheader(f"Average sales for each truck: ${average_sales_2022:.2f}")


        # FOR COMPARISON WITH 2021 DATA
        st.header(f"Comparison with 2021 data")
        # Convert the 'Date' column to datetime format
        df_WO2022['DATE'] = pd.to_datetime(df_WO2022['DATE'])
        df_2021 = df_WO2022[df_WO2022['DATE'].dt.year == 2021]

        filter_rows_2021 = []
        if truckb_input not in df_2021['TRUCK_BRAND_NAME'].values:
            st.error(f"Truck brand name '{truckb_input}' not found in the DataFrame.")
        else:
            for index, row in df_2021.iterrows():
                if (season_input in row['SEASON']) and (truckb_input in row['TRUCK_BRAND_NAME']):
                    filter_rows_2021.append(row)
          
        filter_rows_2021 = pd.DataFrame(filter_rows_2021, columns=df_2021.columns)
        
        # st.write(filter_rows_2021)
        total_sales_of_trucks_2021 = 0
        truck_avail_2021 = filter_rows_2021['TRUCK_ID'].unique()

        # Create a list to hold truck information
        truck_info_2021 = []
        
        for truck in truck_avail_2021:
            total_sales_2021 = filter_rows_2021[filter_rows_2021['TRUCK_ID'] == truck]['TOTAL_SALES_PER_ITEM'].sum()
            truck_info_2021.append({'Truck': truck, 'Total Sales': total_sales_2021})
        
        # Display truck information in a table
        st.table(pd.DataFrame(truck_info_2021))
        
        # Calculate total and average sales
        total_sales_of_trucks_2021 = sum(info['Total Sales'] for info in truck_info_2021)
        average_sales_2021 = total_sales_of_trucks_2021 / len(truck_avail_2021)
        
        # Print total sales for all trucks combined
        st.subheader(f"Total sales for all {len(truck_avail_2021)} trucks: ${total_sales_of_trucks_2021:.2f}")
        
        # Display average sales
        st.subheader(f"Average sales for each truck: ${average_sales_2021:.2f}")

        st.header("Breakdown of Cost for Buying a Food Truck 💸🚚")
        truck_cost = 50000
        operating_costs = 1500
        equipment_costs = 10000
        liscenses_permit = 28000
        other_costs = 2000
        filter_rows_2022 ['cog'] = filter_rows_2022 ['TOTAL_QTY_SOLD'] * filter_rows_2022 ['COG_PER_ITEM_USD']
        average_cog = (filter_rows_2022 ['cog'].sum())/5
        total_cost = truck_cost + operating_costs + equipment_costs + liscenses_permit + other_costs + average_cog

        st.write(f"Food Truck Cost: ${truck_cost}")
        st.write(f"Operating Costs: ${operating_costs} per month")
        st.write(f"Equipment Costs: ${equipment_costs}")
        st.write(f"Equipment Costs: ${equipment_costs}")
        st.write(f"Licenses and Permit Costs: ${liscenses_permit}")
        st.write(f"Average Costs of Goods: ${average_cog}")
        st.write(f"Other Costs: ${other_costs}")
    
        st.subheader("Total Cost: ${:.2f}".format(total_cost))
    


with tab5:
  #Tab 5 code here
  st.write("Hello")



