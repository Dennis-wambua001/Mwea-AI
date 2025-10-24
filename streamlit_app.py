# streamlit_app.py
"""
AI-Based Rice Growth & Yield Monitoring System — Streamlit dashboard
Layout:
- Sidebar: logo, controls (maps toggle, Run Forecasting)
- Main (stacked): Folium map -> Forecast table -> Line chart -> Bar chart
"""

import os
import io
import base64
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import folium
import rasterio
import geopandas as gpd
import plotly.express as px
from matplotlib import cm
from PIL import Image
from streamlit_folium import st_folium
from traitlets import This

# ---------------------- CONFIG / PATHS ----------------------
# NOTE: these are the folders you gave. We normalize paths to avoid stray-space issues.
BASE_DIR ="D:\ Documents\GIS practicals\GIS Prac\MWEA\AI yield system"
CSV_DIR = os.path.normpath(os.path.join(BASE_DIR, "CSV"))
RASTER_DIR = os.path.normpath(os.path.join(BASE_DIR, "Raster"))
SHAPEFILE_DIR = os.path.normpath(os.path.join(BASE_DIR, "Shapefiles"))
ASSETS_DIR = os.path.normpath(os.path.join(BASE_DIR, "Assets"))
LOGO_PATH = os.path.join(ASSETS_DIR, "Dw.png")

FORECAST_CSV = os.path.join("outputs", "forecast_to_2030.csv")
METRICS_CSV = os.path.join("outputs", "model_metrics.csv")
LOCAL_CSV = os.path.join(CSV_DIR, "Mwea data.csv")  # expected CSV filename

# ---------------------- PAGE SETUP ----------------------
st.set_page_config(
    page_title="Mwea Rice Growth & Yield AI Dashboard",
    page_icon="D:\ Documents\GIS practicals\GIS Prac\MWEA\AI yield system\Assets\Dw.png",
    layout="wide",
)

# ---------------------- SIDEBAR (left) ----------------------
with st.sidebar:
    # Logo
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    st.markdown("### 🌾 Mwea Rice Monitoring System")
    st.markdown("AI-Powered Yield Forecasting up to 2030")
    st.markdown("---")

    # Map controls
    st.markdown("#### Map Layers & Controls")
    show_map_btn = st.checkbox("Show Map", value=True)

    # Raster layer checkboxes will be created later inside main map code (if needed).
    st.markdown("---")

    # Run forecasting button
    st.markdown("### Forecasting")
    run_forecast = st.button("▶️ Run Forecasting Pipeline")
    if run_forecast:
        st.info("Running forecasting pipeline — this may take a few minutes...")
        # Run external script (assumes train_and_forecast.py is in same folder and runnable)
        try:
            result = subprocess.run(
                ["python", "train_and_forecast.py"],
                capture_output=True,
                text=True,
                cwd=BASE_DIR  # ensure script runs with project base as cwd
            )
            if result.returncode == 0:
                st.success("Forecasting finished successfully.")
                st.experimental_rerun()  # reload app so forecast CSV appears
            else:
                st.error("Forecasting failed. See console output below.")
                st.text(result.stderr)
        except Exception as e:
            st.error(f"Failed to run forecasting script: {e}")

    st.markdown("---")
    st.markdown("#### 📂 Data Folders")
    st.write("• CSV")
    st.write("• Raster")
    st.write("• Shapefiles")

# ---------------------- MAIN (right) ----------------------
st.title("AI-Based Rice Growth & Yield Monitoring — Mwea Irrigation Scheme")
st.markdown(
    """
    This project develops an artificial intelligence (AI)-driven geospatial system 
    that uses satellite imagery and environmental data to monitor rice growth stages and predict 
    yield across the Mwea Irrigation Scheme. 
    It integrates remote sensing (RS), geographic information systems (GIS), and machine learning (ML) 
    to analyze vegetation indices (NDVI, EVI, NDWI), rainfall, soil and topography 
    to estimate crop performance and spatial yield variability.
    """
)
st.markdown("---")

# Create main single-column stack (map -> forecast table -> charts)
# Use columns to give map more width visually if wanted; here we keep full width, stacked vertically.
# If you prefer map left + charts right, change to st.columns and wrap blocks accordingly.

# ---------------------- MAP SECTION ----------------------
if show_map_btn:
    st.header("🗺️ Map: Raster & Shapefile Layers")
    # build map
    fmap = folium.Map(location=[-0.8, 37.45], zoom_start=10, tiles="CartoDB positron")
    last_bounds = None

    # list raster files
    raster_files = []
    if os.path.exists(RASTER_DIR):
        raster_files = [f for f in os.listdir(RASTER_DIR) if f.lower().endswith(".tif")]
    shapefiles = []
    if os.path.exists(SHAPEFILE_DIR):
        shapefiles = [f for f in os.listdir(SHAPEFILE_DIR) if f.lower().endswith(".shp")]

    # Add rasters as toggleable FeatureGroup layers
    def add_raster(path, layer_name, cmap_name="YlGn", categorical=False):
        global last_bounds
        try:
            with rasterio.open(path) as src:
                arr = src.read(1).astype(float)
                # handle common nodata
                if src.nodata is not None:
                    arr[arr == src.nodata] = np.nan
                arr[arr <= -9999] = np.nan

                # Downsample for quick rendering
                if arr.shape[0] > 800 or arr.shape[1] > 800:
                    arr = arr[::5, ::5]

                if categorical:
                    # simple LUT approach for LULC (fallback if categories > known)
                    unique_vals = np.unique(arr[~np.isnan(arr)])
                    # map categories to tab20 colors
                    cmap = cm.get_cmap("tab20", max(len(unique_vals), 8))
                    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
                    for i, val in enumerate(unique_vals):
                        mask = arr == val
                        color = (np.array(cmap(i)) * 255).astype(np.uint8)
                        rgba[mask] = color
                else:
                    # continuous: stretch 2-98 percentile
                    arr_min, arr_max = np.nanpercentile(arr, [2, 98])
                    arr_norm = (arr - arr_min) / (arr_max - arr_min + 1e-9)
                    arr_norm = np.clip(arr_norm, 0, 1)
                    cmap = cm.get_cmap(cmap_name)
                    rgba = (cmap(arr_norm) * 255).astype(np.uint8)

                # Convert to PNG and add as ImageOverlay under a FeatureGroup
                img = Image.fromarray(rgba)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                img_url = f"data:image/png;base64,{b64}"

                bounds = [[src.bounds.bottom, src.bounds.left],
                          [src.bounds.top, src.bounds.right]]
                last_bounds = bounds

                fg = folium.FeatureGroup(name=layer_name)
                folium.raster_layers.ImageOverlay(
                    image=img_url,
                    bounds=bounds,
                    opacity=0.7,
                    name=layer_name
                ).add_to(fg)
                fg.add_to(fmap)
        except Exception as e:
            st.warning(f"Could not load raster {layer_name}: {e}")

    # Add raster layers with symbology rules
    for rf in raster_files:
        rf_path = os.path.join(RASTER_DIR, rf)
        ln = rf
        low = rf.lower()
        if "ndvi" in low or "evi" in low:
            add_raster(rf_path, ln, cmap_name="YlGn", categorical=False)
        elif "ndwi" in low or "ndwi" in rf.lower():
            add_raster(rf_path, ln, cmap_name="Blues", categorical=False)
        elif "lulc" in low or "landuse" in low:
            add_raster(rf_path, ln, cmap_name="tab20", categorical=True)
        else:
            add_raster(rf_path, ln, cmap_name="viridis", categorical=False)

    # Add shapefile layers as FeatureGroups
    for shp in shapefiles:
        shp_path = os.path.join(SHAPEFILE_DIR, shp)
        try:
            gdf = gpd.read_file(shp_path)
            if gdf.crs and getattr(gdf.crs, "to_epsg", lambda: None)() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            # style choice
            name_lower = shp.lower()
            if "river" in name_lower:
                color = "blue"
            elif "road" in name_lower:
                color = "brown"
            elif "mwea" in name_lower or "boundary" in name_lower:
                color = "black"
            else:
                color = "green"
            fg = folium.FeatureGroup(name=shp.replace(".shp", ""))
            folium.GeoJson(
                gdf,
                style_function=lambda feat, color=color: {"color": color, "weight": 1.5},
                tooltip=folium.GeoJsonTooltip(fields=list(gdf.columns)[:1])
            ).add_to(fg)
            fg.add_to(fmap)
        except Exception as e:
            st.warning(f"Could not load shapefile {shp}: {e}")

    folium.LayerControl(collapsed=False).add_to(fmap)
    if last_bounds:
        fmap.fit_bounds(last_bounds)
    st_data = st_folium(fmap, width=1000, height=550)

st.markdown("---")

# ---------------------- FORECAST TABLE SECTION ----------------------
st.header("📈 Forecasting Results")
if os.path.exists(FORECAST_CSV):
    try:
        df_forecast = pd.read_csv(FORECAST_CSV, parse_dates=["forecast_date"])
        st.success(f"Loaded forecast results ({len(df_forecast)} rows).")
        st.dataframe(df_forecast)  # interactive table
    except Exception as e:
        st.error(f"Failed to load forecast CSV: {e}")
else:
    st.info("No forecast available yet. Click 'Run Forecasting Pipeline' in the sidebar to generate forecasts.")

st.markdown("---")

# ---------------------- CHARTS SECTION ----------------------
st.header("📊 Yield & Rainfall Charts")

# Load main CSV for historical charts
if os.path.exists(LOCAL_CSV):
    try:
        df = pd.read_csv(LOCAL_CSV)
        # Ensure Year numeric and date for plotting
        if "Year" in df.columns:
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
    except Exception as e:
        st.error(f"Failed to load main CSV: {e}")
        df = None
else:
    st.warning("Main CSV not found in CSV folder.")
    df = None

# Combine historical + forecast for line chart if forecast exists
if df is not None:
    # Annual average historical
    hist_line = df.groupby("Year", as_index=False)["Rice_Yield_tonnes"].mean() if "Rice_Yield_tonnes" in df.columns else None

    # If forecast loaded, prepare forecast yearly aggregates
    if os.path.exists(FORECAST_CSV):
        try:
            df_fore = pd.read_csv(FORECAST_CSV, parse_dates=["forecast_date"])
            df_fore["Year"] = df_fore["forecast_date"].dt.year
            fore_line = df_fore.groupby("Year", as_index=False)["forecast_tonnes"].mean()
            # merge to show combined series
            combined_line = pd.concat([
                hist_line.rename(columns={"Rice_Yield_tonnes": "Yield"}),
                fore_line.rename(columns={"forecast_tonnes": "Yield"})], ignore_index=True, sort=False).fillna(method="ffill")
        except Exception:
            combined_line = hist_line.rename(columns={"Rice_Yield_tonnes": "Yield"})
    else:
        combined_line = hist_line.rename(columns={"Rice_Yield_tonnes": "Yield"})

    # Line chart: annual yield (historic + forecast)
    if combined_line is not None and not combined_line.empty:
        fig_line = px.line(combined_line.sort_values("Year"), x="Year", y="Yield",
                           markers=True, title="Rice Yield — Historical + Forecast (Tonnes)",
                           labels={"Year": "Year", "Yield": "Rice Yield (Tonnes)"})
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Not enough data for line chart.")

    # Bar chart: rainfall vs rice yield per year (use historical if available)
    if "Rainfall" in df.columns and "Rice_Yield_tonnes" in df.columns:
        df_bar = df.groupby("Year", as_index=False)[["Rainfall", "Rice_Yield_tonnes"]].mean()
        fig_bar = px.bar(df_bar.sort_values("Year"),
                         x="Year",
                         y=["Rice_Yield_tonnes", "Rainfall"],
                         barmode="group",
                         title="Average Rainfall vs Average Rice Yield (per Year)",
                         labels={"value": "Value", "variable": "Metric"})
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Rainfall or Rice_Yield_tonnes column missing in main CSV — bar chart unavailable.")
else:
    st.info("No historical CSV loaded; charts require CSV data.")
