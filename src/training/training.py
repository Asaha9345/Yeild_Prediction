import pandas as pd
import ee
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from data_collection.data_collection import DataCollection

class Training():

    def make_training_from_df(self, year, df: pd.DataFrame):

        start_date = f'{year}-06-01'
        end_date = f'{year}-09-16'

        df = df.rename(columns={
            'Latitude': 'lat',
            'Longitude': 'lon',
            'Estimated /Observed Yield 2023': 'yield'
        })
        df = df[['lat', 'lon', 'yield']].dropna()
        points = [ee.Geometry.Point(lon, lat) for lat, lon in zip(df['lat'], df['lon'])]
        data_col = DataCollection()
        ndvi_mean, ndvi_std = data_col.get_ndvi(start_date, end_date)
        modis_lst = data_col.get_lst(start_date, end_date)
        rainfall = data_col.get_rainfall(start_date, end_date)
        soil = data_col.get_soil_data()

        def extract_features(pt):
            data = ee.Image.cat([
                ndvi_mean, ndvi_std, modis_lst, rainfall, soil
            ])
            values = data.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=pt,
                scale=30,
                maxPixels=1e9
            )
            return values
        
        features_list = [extract_features(p) for p in points]
        features_info = ee.FeatureCollection([
            ee.Feature(pt, extract_features(pt)) for pt in points
        ])

        # Convert to client-side pandas DataFrame
        features = pd.DataFrame(features_info.getInfo()['features'])
        rows = []
        for f in features_info.getInfo()['features']:
            props = f['properties']
            rows.append(props)
        gee_df = pd.DataFrame(rows)

        # Merge with field data
        full_df = pd.concat([df.reset_index(drop=True), gee_df.reset_index(drop=True)], axis=1)
        full_df = full_df.dropna()

        X = full_df.drop(columns=['yield','lat','lon'])
        y = full_df['yield']

        # Train ML Model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print("R²:", r2_score(y_test, y_pred))
        print("MAE:", mean_absolute_error(y_test, y_pred))

        return model