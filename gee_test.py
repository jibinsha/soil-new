import ee

ee.Initialize(project='soil-nutrient-ai')

def get_ndvi(lat, lon):
    point = ee.Geometry.Point([lon, lat])

    collection = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
        .filterBounds(point) \
        .filterDate('2023-01-01', '2023-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .select(['B4', 'B8'])

    image = collection.median()

    ndvi = image.normalizedDifference(['B8', 'B4'])

    value = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=10
    )

    return value.getInfo()['nd']

# Test
print(get_ndvi(10.5, 76.2))