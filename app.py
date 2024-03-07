from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('/app/model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    latitude = pd.to_numeric(request.form.get('latitude'))
    longitude = pd.to_numeric(request.form.get('longitude'))
    wind = pd.to_numeric(request.form.get('wind'))
    wind_degree = pd.to_numeric(request.form.get('wind_degree'))
    pressure = pd.to_numeric(request.form.get("pressure"))
    precipitation = pd.to_numeric(request.form.get("precipitation"))
    humidity = pd.to_numeric(request.form.get("humidity"))
    cloud = pd.to_numeric(request.form.get("cloud"))
    temperature = "temperature"
    input_query = np.array([[latitude, longitude, wind, wind_degree, pressure, precipitation, humidity, cloud]])
    result = model.predict(input_query)[0]
    return jsonify({'temperature' : str(result)})

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
