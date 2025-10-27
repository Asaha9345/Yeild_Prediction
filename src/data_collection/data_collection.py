import ee

class DataCollection:

    def get_ndvi(self, start_date, end_date):
        # Step 1: Define GEE Datasets ------------------------------------
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .select(['B2', 'B3', 'B4', 'B8'])
        def add_ndvi(img):
            return img.addBands(img.normalizedDifference(['B8', 'B4']).rename('NDVI'))
        ndvi = s2.map(add_ndvi)
        ndvi_mean = ndvi.select('NDVI').mean().rename('NDVI')
        ndvi_std = ndvi.select('NDVI').reduce(ee.Reducer.stdDev()).rename('NDVI_stdDev')

        return ndvi_mean, ndvi_std
    
    def get_lst(self, start_date, end_date):

        # MODIS LST
        modis_lst = ee.ImageCollection('MODIS/061/MOD11A2') \
            .filterDate(start_date, end_date) \
            .select('LST_Day_1km') \
            .mean().multiply(0.02).rename('LST_Day_1km')
        
        return modis_lst
    
    def get_rainfall(self, start_date, end_date):

        # Rainfall (CHIRPS)
        rainfall = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
            .filterDate(start_date, end_date) \
            .select('precipitation') \
            .sum().rename('precipitation')
        
        return rainfall
    
    def get_soil_data(self):

        # Soil organic carbon (OpenLandMap)
        soil = ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02") \
            .select('b0').rename('Soil_OC')
        
        return soil