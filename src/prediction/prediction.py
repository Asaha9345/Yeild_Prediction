import joblib
import json
import ee
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from data_collection.data_collection import DataCollection

class Prediction:

    def __init__(self, aoi:gpd.GeoDataFrame , model_path: str):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_full_path = os.path.join(base_dir, model_path)
        if not os.path.exists(model_full_path):
            raise FileNotFoundError(f"Model not found at {model_full_path}")
        self.model = joblib.load(model_full_path)
        self.aoi = aoi

    def make_prediction(self, year):

        def get_agrimask(roi):
            am1 = (ee.ImageCollection("ESA/WorldCover/v100")
                .first()
                .eq(40)
                .clip(roi)
                .selfMask())
            am2 = (ee.ImageCollection("ESA/WorldCover/v200")
                .first()
                .eq(40)
                .clip(roi)
                .selfMask())
            am = am1.Or(am2)
            return am
        roi = ee.FeatureCollection(ee.FeatureCollection(json.loads(self.aoi.to_json())))
        agrimask = get_agrimask(roi)
        start_date = f"{year}-06-01"
        end_date = f"{year}-09-16"
        data_col = DataCollection()
        ndvi_mean, ndvi_std = data_col.get_ndvi(start_date, end_date)
        modis_lst = data_col.get_lst(start_date, end_date)
        rainfall = data_col.get_rainfall(start_date, end_date)
        soil = data_col.get_soil_data()
        data_img = ee.Image.cat([ndvi_mean.updateMask(agrimask), ndvi_std.updateMask(agrimask), 
                                 modis_lst.updateMask(agrimask), rainfall, soil.updateMask(agrimask)])
        def feature_from_row(row):
            geom = ee.Geometry.Polygon(row.geometry.__geo_interface__["coordinates"])
            return ee.Feature(geom)
        
        gee_features = [feature_from_row(row) for _, row in self.aoi.iterrows()]

        def extract_mean_values(feat):
            stats = data_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=feat.geometry(),
                scale=30,
                maxPixels=1e13
            )
            return ee.Feature(feat.geometry(), stats)

        extracted = ee.FeatureCollection([extract_mean_values(f) for f in gee_features])
        extracted_info = extracted.getInfo()

        # Step 4: Convert to DataFrame ----------------------------------
        rows = []
        for f in tqdm(extracted_info['features'], desc="Extracting GEE features"):
            props = f['properties']
            rows.append(props)

        feature_df = pd.DataFrame(rows)

        # Step 5: Prepare for prediction --------------------------------
        expected_features = ['LST_Day_1km', 'NDVI', 'NDVI_stdDev', 'Soil_OC', 'precipitation']
        for col in expected_features:
            if col not in feature_df.columns:
                feature_df[col] = None

        feature_df = feature_df[expected_features].fillna(feature_df.mean())

        # Predict
        predicted_yield = self.model.predict(feature_df)

        # Step 6: Combine with original GeoDataFrame --------------------
        gdf = self.aoi.copy()
        gdf['predicted_yield'] = predicted_yield

        return gdf

