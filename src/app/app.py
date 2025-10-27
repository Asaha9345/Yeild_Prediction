import streamlit as st
import geopandas as gpd
import geemap.foliumap as geemap
import ee
import json
from prediction.prediction import Prediction
from auth.authenticate import initialize_gee
from shapely.geometry import Point
import folium
import plotly.graph_objects as go
import os

class YieldPredictionApp:
    """Streamlit app for district-wise kharif paddy yield prediction."""

    def __init__(self):
        """Initialize app settings and load data."""
        st.set_page_config(
            page_title="Yield Prediction",
            page_icon="🌾",
            layout="wide"
        )
        self.data_path = "https://drive.google.com/file/d/1QKRfJ_JfnDMILRVxG0BbhVvJ7H7CxPwr/view?usp=sharing"
        self.shapefile = None
        self.state = None
        self.selected_districts = []
        self.year = None

        initialize_gee()
        self.load_data()

    def load_data(self):
        """Load district shapefile data."""
        try:
            self.shapefile = gpd.read_file(self.data_path)
        except Exception as e:
            st.error(f"Error loading shapefile: {e}")
            st.stop()

    def sidebar(self):
        """Render sidebar controls."""
        st.sidebar.header("Navigation")

        states = sorted(list(self.shapefile["State"].unique()))
        self.state = st.sidebar.selectbox("Choose a State", states)

        if self.state:
            filtered = self.shapefile[self.shapefile["State"] == self.state]
            districts = sorted(list(filtered["District"].unique()))
            self.selected_districts = st.sidebar.selectbox("Choose District", districts)
            self.year = st.sidebar.number_input("Select Prediction Year", min_value=2023)

            if self.selected_districts and self.year:
                if st.sidebar.button("Run Prediction"):
                    self.display_map(filtered)

    def display_map(self, filtered_shapefile):
    
        """Display selected area on map with yield labels and visible AOI boundary."""

        # Filter by district
        shapefile_subset = filtered_shapefile[
            filtered_shapefile["District"] == self.selected_districts
        ]

        # Run prediction
        predict_class = Prediction(shapefile_subset, r"model/gee_yield_model.pkl")
        predicted_shapefile = predict_class.make_prediction(self.year)

        # Convert to EE FeatureCollection
        aoi = ee.FeatureCollection(json.loads(predicted_shapefile.to_json()))

        # Create the map
        Map = geemap.Map()
        Map.add_basemap("HYBRID")

        # AOI as hollow polygons
        outline_style = {
            'color': 'yellow',
            'fillColor': '00000000',  # transparent
            'width': 2
        }
        Map.addLayer(aoi.style(**outline_style), {}, "AOI Boundary")
        Map.centerObject(aoi)

        # Add dynamic text labels (interactive)
        for _, row in predicted_shapefile.iterrows():
            centroid = row.geometry.centroid
            label_text = f"{row['predicted_yield']:.2f} t/ha"

            # Use HTML + CSS so the text scales with zoom
            html = f"""
            <div style="
                font-size: 12px;
                color: #00ff00;
                font-weight: bold;
                text-shadow: 1px 1px 2px black;
                white-space: nowrap;
            ">{label_text}</div>
            """

            # Create DivIcon for interactive text
            folium.Marker(
                location=[centroid.y, centroid.x],
                icon=folium.DivIcon(html=html)
            ).add_to(Map)

        # Send map to Streamlit
        Map.to_streamlit()

        # Detect Streamlit theme (light or dark)
        theme = st.get_option("theme.base")  # returns "light" or "dark"

        # Define colors based on theme
        if theme == "dark":
            bg_color = "#000000"
            text_color = "#FFFFFF"
            bar_color = "#FFFFFF"
        else:
            bg_color = "#FFFFFF"
            text_color = "#000000"
            bar_color = "#000000"

        # Create figure
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Block wise Predicted Yield",
            x=list(predicted_shapefile["Block"]),
            y=list(predicted_shapefile["predicted_yield"]),
            marker_color=bar_color,
            opacity=0.8
        ))

        # Update layout
        fig.update_layout(
            title=dict(
                text="Distribution of Yield Across Blocks",
                x=0.5,
                xanchor="center",
                font=dict(size=22, color=text_color)
            ),
            xaxis=dict(
                title=dict(text="Blocks", font=dict(size=15, color=text_color)),
                tickfont=dict(size=13, color=text_color),
                showgrid=False,
                linecolor=text_color
            ),
            yaxis=dict(
                title=dict(text="Crop Yield (t/ha)", font=dict(size=15, color=text_color)),
                tickfont=dict(size=13, color=text_color),
                showgrid=False,
                linecolor=text_color
            ),
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color),
            legend=dict(
                title=dict(text="Legend", font=dict(size=16, color=text_color)),
                x=0.95,
                y=1,
                xanchor="left",
                yanchor="top",
                bgcolor=bg_color,
                bordercolor=text_color,
                borderwidth=1,
                font=dict(size=13, color=text_color),
                orientation="v"
            ),
            margin=dict(l=60, r=60, t=70, b=60),
        )

        # Add a subtle animation-like hover effect
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Yield: %{y:.2f} t/ha<extra></extra>",
        )

        # Display chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)




    def main(self):
        """Main function to run the app."""
        st.title("District-Wise Kharif Paddy Yield Prediction")
        self.sidebar()

        st.markdown("---")
        st.caption("Made by Prajukta Research")


# if __name__ == "__main__":
#     app = YieldPredictionApp()
#     app.main()
