import ee
import streamlit as st

def initialize_gee():
    """Initialize GEE using credentials from Streamlit's secrets.toml."""
    try:
        # ✅ Streamlit automatically loads `.streamlit/secrets.toml`
        service_account = st.secrets["client_email"]
        private_key = st.secrets["private_key"]
        project_id = st.secrets["project_id"]

        credentials = ee.ServiceAccountCredentials(service_account, key_data=private_key)
        ee.Initialize(credentials, project=project_id)
        print("Google Earth Engine initialized successfully!")
    except Exception as e:
        print(f"GEE initialization failed: {e}")
