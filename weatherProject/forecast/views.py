from django.shortcuts import render
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pytz
import os

API_KEY = '3d36fa5ac8f729866b7cf57f323afaac'  # Replace with your actual API key
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# 1. Fetch Current Weather Data
def get_current_weather(city):
    try:
        url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if data.get("cod") != 200:
            raise ValueError(f"Error: {data.get('message', 'Unknown error occurred')}")

        return {
            'city': data['name'],
            'current_temp': round(data['main']['temp'], 1),
            'feels_like': round(data['main']['feels_like'], 1),
            'temp_min': round(data['main']['temp_min']),
            'temp_max': round(data['main']['temp_max']),
            'humidity': round(data['main']['humidity'], 1),
            'description': data['weather'][0]['description'],
            'country': data['sys']['country'],
            'wind_gust_dir': data['wind']['deg'],
            'pressure': data['main']['pressure'],
            'Wind_Gust_Speed': data['wind']['speed'],
            'clouds': data['clouds']['all'],
            'Visibility': data['visibility'],
        }
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to retrieve weather data: {e}")

# 2. Read Historical Data
def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna()
    df = df.drop_duplicates()
    return df

# 3. Prepare Data for Training
def prepare_data(data):
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

    X = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    y = data['RainTomorrow']
    return X, y, le

# 4. Train Rain Prediction Model
def train_rain_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train) #train the model

    y_pred = model.predict(X_test) #to make prediction on test set
    print("Mean Squared Error for Rain Model:", mean_squared_error(y_test, y_pred))

    return model

# 5. Prepare Regression Data
def prepare_regression_data(data, feature):
    X, y = [], []
    feature_values = data[feature].values

    for i in range(len(feature_values) - 1):
        X.append(feature_values[i])
        y.append(feature_values[i + 1])

    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    return X, y

# 6. Train Regression Model
def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# 7. Predict Future
def predict_future(model, current_value):
    predictions = [current_value]
    for i in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(round(next_value[0], 1))

    return predictions[1:]

# 8. Weather Analysis Function
def weather_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')

        # Check if city is provided and contains only alphabets (valid city name)
        if not city or not city.replace(" ", "").isalpha():
            context = {
                'error_message': 'Invalid city name! Please enter a valid city name containing only alphabets.',
            }
            return render(request, 'weather.html', context)

        try:
            # Attempt to fetch weather data
            current_weather = get_current_weather(city)
        except ValueError as e:
            context = {'error_message': str(e)}
            return render(request, 'weather.html', context)

        try:
            # Load historical data
            csv_path = os.path.join('D:\\Oakland University\\Fall semester\\Software Engineering\\SEMachineLearningProject\\weatherProject\\weather.csv')
            historical_data = read_historical_data(csv_path)

            # Prepare and train the rain prediction model
            X, y, le = prepare_data(historical_data)
            rain_model = train_rain_model(X, y)

            # Map wind direction to compass points
            wind_deg = current_weather['wind_gust_dir'] % 360
            compass_points = [
                ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
                ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
                ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
                ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
                ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
                ("NNW", 326.25, 348.75)
            ]
            compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)

            # Handle unknown compass direction encoding
            compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1
            current_data = {
                'MinTemp': current_weather['temp_min'],
                'MaxTemp': current_weather['temp_max'],
                'WindGustDir': compass_direction_encoded,
                'WindGustSpeed': current_weather['Wind_Gust_Speed'],
                'Humidity': current_weather['humidity'],
                'Pressure': current_weather['pressure'],
                'Temp': current_weather['current_temp'],
            }

            current_df = pd.DataFrame([current_data])

            # Rain prediction
            rain_prediction = rain_model.predict(current_df)[0]

            # Prepare regression models for temperature and humidity
            X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
            X_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')

            temp_model = train_regression_model(X_temp, y_temp)
            hum_model = train_regression_model(X_hum, y_hum)

            # Predict future temperature and humidity
            future_temp = predict_future(temp_model, current_weather['temp_min'])
            future_humidity = predict_future(hum_model, current_weather['humidity'])

            # Prepare time for future predictions
            timezone = pytz.timezone('Asia/Karachi')
            now = datetime.now(timezone)
            next_hour = now + timedelta(hours=1)
            next_hour = next_hour.replace(minute=0, second=0, microsecond=0)

            future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

            # Store each value separately
            time1, time2, time3, time4, time5 = future_times
            temp1, temp2, temp3, temp4, temp5 = future_temp
            hum1, hum2, hum3, hum4, hum5 = future_humidity

            context = {
                'location' : city,
                'country': current_weather['country'],
                'description': current_weather['description'],
                'current_temp': current_weather['current_temp'],
                'feels_like': current_weather['feels_like'],
                'MinTemp': current_weather['temp_min'],
                'MaxTemp': current_weather['temp_max'],
                'humidity': current_weather['humidity'],
                'pressure': current_weather['pressure'],
                'clouds': current_weather['clouds'],
                'city' : current_weather['city'],
                'visibility': current_weather['Visibility'],
                
                'wind': current_weather['Wind_Gust_Speed'],
                'rain_prediction': "Yes" if rain_prediction == 1 else "No",

                'time': datetime.now(),
                'date': datetime.now().strftime("%B %d, %Y"),

                'time1': time1, 
                'time2': time2, 
                'time3': time3, 
                'time4': time4, 
                'time5': time5,

                'temp1': temp1, 
                'temp2': temp2, 
                'temp3': temp3, 
                'temp4': temp4, 
                'temp5': temp5,

                'hum1': hum1, 
                'hum2': hum2, 
                'hum3': hum3, 
                'hum4': hum4, 
                'hum5': hum5,
            }

            return render(request, 'weather.html', context)

        except ValueError as e:
            context = {'error_message': str(e)}
            return render(request, 'weather.html', context)

    # If GET request, render an empty form
    return render(request, 'weather.html')
