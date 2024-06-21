from modules.ChatCompletions import get_chat_completions
import pandas as pd  # Importing the pandas library for data manipulation
import json

def updateHotelsCSV():
    hotels_df = pd.read_csv('data/final11.csv', encoding='unicode_escape')
    #iterate_llm_response(product_map_layer, hotel_description_1)
    
    ## Create a new column "hotel_preference" that contains the dictionary of the hotel features
    hotels_df['hotel_preference'] = hotels_df['Description'].apply(lambda x: product_map_layer(x))
    print(hotels_df)
    hotels_df.to_csv("updated_hotels.csv",index=False,header = True)


def product_map_layer(hotel_description):
    delimiter = "#####"

    hotel_spec = {
        "City":"(Actual location of the hotel)",
        "RestaurantType":"either one are all of 'Casual Dining'/'QuickBites'/'Buffet'",
        "ApproximateCost":"the approximate rate in INR",
        "Cuisine":"either one are all of 'North Indian'/'South Indian'/'Chinese'",
        "Votes":"high/medium/low"
    }

    #values = {'low','medium','high'}

    prompt=f"""
    You are a Hotel Preference suggestion guide whose job is to extract the key features of hotel and classify them as per their requirements.
    To analyze each hotel, perform the following steps:
    Step 1: Extract the hotel's location and all other details from the description {hotel_description}
    Step 2: Store the extracted features in {hotel_spec} for each hotel \
    Step 3: Classify each of the items in {hotel_spec} into corresponding values based on the following rules: \
    {delimiter}
    City:
    - <<< Should be the actual city/location defined in the description. Should be 1 or 2 words only >>> , \n

    RestaurantType:
    - <<< Should be either or one of these values 'Casual Dining'/'QuickBites'/'Buffet'. Make sure to build a list of objects if the hotel satisfies more than one. \n

    ApproximateCost:
    - <<< Should be a numerical without any currency detail retrieved from the hotel description >>>

    Cuisine:
    - <<< Should be either or one of these values 'North Indian'/'South Indian'/'Chinese'. Make sure to build a list of objects if the hotel satisfies more than one cuisine. >>> , \n
    
    Votes:
    - low: <<< if votes less than 300 >>> , \n
    - medium: <<< if votes between 300 and 700 >>> , \n
    - high: <<< if votes greater than 700 >>> \n
    {delimiter}

    {delimiter}
    Here is input output pair for few-shot learning:
    input 1: "Experience a delightful culinary journey at our Buffet restaurant located in the vibrant Banashankari area. With an approximate cost of _800 for two people, indulge in a variety of exquisite cuisines including North Indian, Mughlai, and Chinese. Our highly rated restaurant, with 775 votes, reflects the satisfaction of our valued customers. Enjoy the convenience of online ordering and the option to book a table to ensure your spot in our popular dining area.
Savor our highly recommended dishes such as Pasta, Lunch Buffet, Masala Papad, Paneer Lajawab, Tomato Shorba, Dum Biryani, and Sweet Corn Soup. Whether you are planning a casual meal or a special gathering, our buffet offers a delectable spread that caters to all tastes."
    output 1: {{ "City": "Banashankari", "RestaurantType": [ "Casual Dining", "special gathering", "Buffet" ], "ApproximateCost": 800, "Cuisine": [ "North Indian, Mughlai, Chinese" ], "Votes": "high" }}

    {delimiter}
    
    Strictly don't keep any values in the JSON dictionary that the system doesn't understand.
    """
    input = f"""Follow the above instructions step-by-step and output the dictionary in JSON format {hotel_spec} for the following laptop {hotel_description}."""

    messages=[{"role": "system", "content":prompt },{"role": "user","content":input}]
    #print(messages)

    response = get_chat_completions(messages, json_format = True)
    print('===========',response,'=============')

    return response


def compare_hotels_with_user(user_preference):
    #user_requirements = { 'City': ['Banashankari','Jayanagar'], 'RestaurantType': 'Casual Dining', 'ApproximateCost': 700, 'Cuisine': 'Chinese', 'Votes': 'medium' }"Bannerghatta Road", "RestaurantType":"Buffet", "ApproximateCost":600, "Cuisine":"North Indian", "Votes":
    #user_preference = '{"City":["Bannerghatta Road"], "RestaurantType":"Buffet", "ApproximateCost":600, "Cuisine":"North Indian", "Votes":"high"}';
    user_requirements = json.loads(user_preference)

    hotels_df = pd.read_csv('data/updated_hotels.csv')
    
    # Extracting user requirements from the input string (assuming it's a dictionary)
    # Since the function parameter already seems to be a string, we'll use it directly instead of extracting from a dictionary
    
    filtered_hotels = hotels_df.copy()
    
    #filter hotels based on location
    city = user_requirements.get('City')
    print(type(city))

    if (isinstance(city, str)):
        filtered_hotels = filtered_hotels[filtered_hotels['listed_in(city)'] == city].copy()
    else:
        filtered_hotels = filtered_hotels[filtered_hotels['listed_in(city)'].isin(city)].copy()
    
    # Extracting the cost value from user_requirements and converting it to an integer
    if (isinstance(user_requirements.get('ApproximateCost'), str)):
        budget = int(user_requirements.get('ApproximateCost', '0').replace(',', '').split()[0])
        #filtered_hotels['approx_cost(for two people)'] = filtered_hotels['approx_cost(for two people)'].str.replace(',', '').astype(int)
    else:
        budget = user_requirements.get('ApproximateCost')


    filtered_hotels['approx_cost(for two people)'] = filtered_hotels['approx_cost(for two people)'].str.replace(',', '').astype(int)
    
    if (budget != 0):
        filtered_hotels = filtered_hotels[filtered_hotels['approx_cost(for two people)'] <= budget].copy()
    
    filtered_hotels['Score'] = 0
    
    for index, row in filtered_hotels.iterrows():
        user_product_match_str = row['hotel_preference']
        hotel_values = user_product_match_str
        hotel_values = { 'City': 'Banashankari', 'RestaurantType': 'Casual Dining', 'ApproximateCost': 700, 'Cuisine': 'Chinese', 'Votes': 'medium' }
        #hotel_values = dictionary_present(user_product_match_str)
        score = 2 #since the location and budget is matched already
        
        # Comparing user requirements with laptop features and updating scores
        for key, user_value in user_requirements.items():
            if key == 'ApproximateCost' or key == 'City':
                continue  # Skipping budget comparison
            hotel_value = hotel_values.get(key, None)
            
            #print(hotel_value, user_value)
            if user_value in hotel_value:
                score += 1  # Incrementing score if laptop value meets or exceeds user value
        
        filtered_hotels.loc[index, 'Score'] = score  # Updating the 'Score' column in the DataFrame

    # Sorting laptops by score in descending order and selecting the top 3 products
    top_hotels = filtered_hotels.drop('hotel_preference', axis=1)
    top_hotels = top_hotels.sort_values('Score', ascending=False).head(3)
    top_hotels_json = top_hotels.to_json(orient='records')  # Converting the top laptops DataFrame to JSON format

    # top_laptops
    return top_hotels_json


def recommendation_validation(hotel_recommendation):
    data = json.loads(hotel_recommendation)
    data1 = []
    for i in range(len(data)):
        if data[i]['Score'] > 2:
            data1.append(data[i])

    return data1

def initialize_conv_reco(hotels):
    system_message = f"""
    You are an intelligent hotel recommendation expert and you are tasked with the objective to \
    solve the user queries about any hotel from the catalogue in the user message \
    You should keep the user profile in mind while answering the questions.\

    Start with a brief summary of each hotel in the following format, in decreasing order of approximate cost of the hotel:
    1. <Hotel Name> : <Address>, <Price in Rs>, <URL>
    2. <Hotel Name> : <Address>, <Price in Rs>, <URL>

    """
    user_message = f""" These are the hotels in which user might have questions: {hotels}"""
    conversation = [{"role": "system", "content": system_message },
                    {"role":"user","content":user_message}]
    # conversation_final = conversation[0]['content']
    return conversation