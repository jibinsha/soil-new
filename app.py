from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import ee
import datetime

app = Flask(__name__)   # ✅ FIRST create app
CORS(app)  

import os
import json
from google.oauth2 import service_account

credentials_dict = json.loads(os.environ.get("GOOGLE_CREDENTIALS"))

credentials = service_account.Credentials.from_service_account_info(
    credentials_dict,
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

ee.Initialize(credentials)

# ✅ Load trained ML model
model = joblib.load("soil_model.pkl")
#//////////
def recommend_crop(data):
    N = data.get("N", 0)
    P = data.get("P", 0)
    K = data.get("K", 0)
    pH = data.get("pH", 7)
    ndvi = data.get("NDVI", 0)

    if pH < 5.5:
        return "Rubber, Tea"
    elif 5.5 <= pH <= 6.5:
        if ndvi > 0.5:
            return "Rice, Banana"
        else:
            return "Groundnut, Pulses"
    elif pH > 6.5:
        return "Coconut, Vegetables"

    return "No clear recommendation"
# =========================
# 🌱 NDVI (Sentinel-2)
# =========================
def get_ndvi(lat, lon):
    point = ee.Geometry.Point([lon, lat])

    today = datetime.date.today()
    start = (today - datetime.timedelta(days=10)).strftime('%Y-%m-%d')
    end = today.strftime('%Y-%m-%d')

    collection = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
        .filterBounds(point) \
        .filterDate(start, end) \
        .sort('system:time_start', False) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .select(['B4', 'B8'])

    image = collection.first()

    ndvi = image.normalizedDifference(['B8', 'B4'])

    value = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=10
    )

    return value.getInfo().get('nd', 0)


# =========================
# 🏔️ Elevation (DEM)
# =========================
def get_elevation(lat, lon):
    point = ee.Geometry.Point([lon, lat])

    dataset = ee.Image("USGS/SRTMGL1_003")

    value = dataset.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=30
    )

    return value.getInfo().get('elevation', 0)
#SLOPE ######
def get_slope(lat, lon):
    point = ee.Geometry.Point([lon, lat])

    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem)

    value = slope.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=30
    )

    return value.getInfo().get('slope', 2)
# =========================
# 🌧️ Rainfall (Last 7 days)
# =========================
def get_rainfall(lat, lon):
    point = ee.Geometry.Point([lon, lat])

    today = datetime.date.today()
    start = (today - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    end = today.strftime('%Y-%m-%d')

    collection = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterBounds(point) \
        .filterDate(start, end)

    rainfall = collection.sum()

    value = rainfall.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=5000
    )
    
    return value.getInfo().get('precipitation', 0)

def get_temperature(lat, lon):
    point = ee.Geometry.Point([lon, lat])

    today = datetime.date.today()
    start = (today - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    end = today.strftime('%Y-%m-%d')

    collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .filterBounds(point) \
        .filterDate(start, end)

    # 🔥 Check if empty
    size = collection.size().getInfo()

    if size == 0:
        # fallback → extend range
        start = (today - datetime.timedelta(days=15)).strftime('%Y-%m-%d')

        collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
            .filterBounds(point) \
            .filterDate(start, end)

    image = collection.sort('system:time_start', False).first()

    # 🔥 STILL safety check
    if image is None:
        return 25, 35  # fallback values

    values = image.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=1000
    ).getInfo()

    temp_min = values.get('temperature_2m_min', 300) - 273.15
    temp_max = values.get('temperature_2m_max', 300) - 273.15

    return temp_min, temp_max
#RAINFALL AND TEM DATA FOR CROP RECOMMENDATION 
def get_annual_rainfall(lat, lon):
    point = ee.Geometry.Point([lon, lat])

    today = datetime.date.today()
    start = (today - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    end = today.strftime('%Y-%m-%d')

    collection = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterBounds(point) \
        .filterDate(start, end)

    rainfall = collection.sum()

    value = rainfall.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=5000
    )

    return value.getInfo().get('annual_rainfall', 0)
def get_mean_temperature(lat, lon):
    point = ee.Geometry.Point([lon, lat])

    today = datetime.date.today()
    start = (today - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    end = today.strftime('%Y-%m-%d')

    collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .filterBounds(point) \
        .filterDate(start, end)

    image = collection.mean()

    values = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=1000
    ).getInfo()

    mean_temperature = values.get('temperature_2m', 300) - 273.15

    return mean_temperature

# CROP RECOMMENDATION
crop_database = [

    {"name": "Rice",
     "annual_rainfall": (1000,2500), "mean_temperature": (20,35),
     "elevation": (0, 500), "slope": (0, 5)},

    {"name": "Maize",
     "annual_rainfall": (500,1200), "mean_temperature": (18,32),
     "elevation": (0, 1000), "slope": (0, 8)},

    {"name": "Groundnut",
     "annual_rainfall": (500,1000), "mean_temperature": (25,35),
     "elevation": (0, 600), "slope": (0, 5)},

    {"name": "Pulses",
     "annual_rainfall": (400,800), "mean_temperature": (20,30),
     "elevation": (0, 800), "slope": (0, 6)},


    {"name": "Banana",
     "annual_rainfall": (1000,3000), "mean_temperature": (20,35),
     "elevation": (0, 1200), "slope": (0, 8)},

    {"name": "Mango",
     "annual_rainfall": (750,2500), "mean_temperature": (24,35),
     "elevation": (0, 1000), "slope": (0, 10)},

    {"name": "Papaya",
     "annual_rainfall": (1000,2000), "mean_temperature": (22,35),
     "elevation": (0, 800), "slope": (0, 6)},

    {"name": "Vegetables",
     "annual_rainfall": (600,1500), "mean_temperature": (20,35),
     "elevation": (0, 1200), "slope": (0, 10)},


    {"name": "Rubber",
     "annual_rainfall": (2000,3500), "mean_temperature": (25,35),
     "elevation": (0, 600), "slope": (0, 15)},

    {"name": "Coffee",
     "annual_rainfall": (1500,2500), "mean_temperature": (18,28),
     "elevation": (600, 1600), "slope": (5, 25)},

    {"name": "Tea",
     "annual_rainfall": (1500,3000), "mean_temperature": (18,25),
     "elevation": (1000, 2200), "slope": (10, 30)},

    {"name": "Cardamom",
     "annual_rainfall": (1500,3000), "mean_temperature": (18,28),
     "elevation": (800, 1600), "slope": (10, 30)},

    {"name": "Black Pepper",
     "annual_rainfall": (2000,3000), "mean_temperature": (23,32),
     "elevation": (0, 1200), "slope": (5, 25)},


    {"name": "Coconut",
     "annual_rainfall": (1000,3000), "mean_temperature": (20,35),
     "elevation": (0, 600), "slope": (0, 8)},

    {"name": "Arecanut",
     "annual_rainfall": (1500,3000), "mean_temperature": (20,35),
     "elevation": (0, 800), "slope": (0, 10)}
]
def score_range(value, low, high):
    if low <= value <= high:
        return 1
    else:
        return max(0, 1 - abs(value - (low+high)/2) / (high-low))


def recommend_crop(data):
    
    rainfall = data.get("annual_rainfall", 2000)
    temp = data.get("mean_temperature", 28)
    elevation = data.get("Elevation", 0)
    slope = data.get("Slope", 5)
    
    results = []

    for crop in crop_database:
        score = 0
        score += score_range(rainfall, *crop["annual_rainfall"])
        score += score_range(temp, *crop["mean_temperature"])
        score += score_range(elevation, *crop["elevation"])
        score += score_range(slope, *crop["slope"])

        results.append({
            "crop": crop["name"],
            "score": round(score,2)
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:4]
# =========================
# 🚀 API ROUTES
# =========================
@app.route('/')
def home():
    return "Geo-AI Soil Intelligence API Running 🚀"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        lat = data['lat']
        lon = data['lon']
    
        # 🔥 Fetch REAL data
        ndvi = get_ndvi(lat, lon)
        elevation = get_elevation(lat, lon)
        rainfall = get_rainfall(lat, lon)
        min_temp, max_temp = get_temperature(lat, lon)
        slope = get_slope(lat, lon)
        annual_rainfall = get_annual_rainfall(lat, lon)
        mean_temp = get_mean_temperature(lat, lon)
        # ⚠️ Temporary assumptions
        humidity = 80

        # Feature array (same order as training)
        features = np.array([[lat, lon, elevation, slope, rainfall,
                              min_temp, max_temp, humidity, ndvi]])

        prediction = model.predict(features)[0]

        labels = ['N','P','K','Ca','Mg','S','Zn','Fe','Mn','Cu','B','pH','EC','Org_C']

        result = {label: float(value) for label, value in zip(labels, prediction)}

        # Add satellite data also in response
        result.update({
            "NDVI": ndvi,
            "Elevation": elevation,
            "Rainfall": rainfall,
            "Min_Temp": min_temp,
            "Max_Temp": max_temp
        })

        crop_input = {
            "annual_rainfall": annual_rainfall,
            "mean_temperature": mean_temp,
            "Elevation": elevation,
            "Slope": slope
        }

        recommendations = recommend_crop(crop_input)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})


# =========================
# ▶️ RUN APP
# =========================
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
