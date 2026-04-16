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
# CROP RECOMMENDATION
crop_database = [

    # 🌾 FIELD CROPS
    {"name": "Rice", "N": (80,150), "P": (40,80), "K": (40,80),
     "rainfall": (1000,2500), "temp": (20,35)},

    {"name": "Maize", "N": (120,200), "P": (50,100), "K": (50,100),
     "rainfall": (500,1200), "temp": (18,32)},

    {"name": "Groundnut", "N": (20,60), "P": (40,80), "K": (40,80),
     "rainfall": (500,1000), "temp": (25,35)},

    {"name": "Pulses", "N": (20,50), "P": (30,60), "K": (30,60),
     "rainfall": (400,800), "temp": (20,30)},


    # 🍌 HORTICULTURE
    {"name": "Banana", "N": (200,300), "P": (60,100), "K": (200,300),
     "rainfall": (1000,3000), "temp": (20,35)},

    {"name": "Mango", "N": (100,200), "P": (50,100), "K": (100,200),
     "rainfall": (750,2500), "temp": (24,35)},

    {"name": "Papaya", "N": (150,250), "P": (60,100), "K": (150,250),
     "rainfall": (1000,2000), "temp": (22,35)},

    {"name": "Vegetables", "N": (100,200), "P": (50,100), "K": (100,200),
     "rainfall": (600,1500), "temp": (20,35)},


    # 🌿 PLANTATION CROPS (KERALA CORE)
    {"name": "Rubber", "N": (50,100), "P": (25,50), "K": (50,100),
     "rainfall": (2000,3500), "temp": (25,35)},

    {"name": "Coffee", "N": (100,150), "P": (50,80), "K": (100,150),
     "rainfall": (1500,2500), "temp": (18,28)},

    {"name": "Tea", "N": (100,200), "P": (40,80), "K": (100,200),
     "rainfall": (1500,3000), "temp": (18,25)},

    {"name": "Cardamom", "N": (75,150), "P": (40,75), "K": (75,150),
     "rainfall": (1500,3000), "temp": (18,28)},

    {"name": "Black Pepper", "N": (100,150), "P": (50,100), "K": (100,150),
     "rainfall": (2000,3000), "temp": (23,32)},


    # 🥥 TREE CROPS
    {"name": "Coconut", "N": (100,200), "P": (40,80), "K": (120,200),
     "rainfall": (1000,3000), "temp": (20,35)},

    {"name": "Arecanut", "N": (100,200), "P": (40,80), "K": (100,200),
     "rainfall": (1500,3000), "temp": (20,35)}
]
def score_range(value, low, high):
    if low <= value <= high:
        return 1
    else:
        return max(0, 1 - abs(value - (low+high)/2) / (high-low))


def recommend_crop(data):
    N = data.get("N", 0)
    P = data.get("P", 0)
    K = data.get("K", 0)
    rainfall = data.get("Rainfall", 0)
    temp = data.get("Max_Temp", 30)

    results = []

    for crop in crop_database:
        score = 0
        score += score_range(N, *crop["N"])
        score += score_range(P, *crop["P"])
        score += score_range(K, *crop["K"])
        score += score_range(rainfall, *crop["rainfall"])
        score += score_range(temp, *crop["temp"])

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

        # ⚠️ Temporary assumptions
        humidity = 80
        slope = 2

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

        recommendations = recommend_crop(result)
        result["Crop_Recommendation"] = recommendations
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
