import folium
import rasterio
import numpy as np
from matplotlib import cm
from PIL import Image
import io, base64

# 👇 CHANGE this to one of your raster files
raster_path ="D:\ Documents\GIS practicals\GIS Prac\MWEA\AI yield system\Raster\NDVI_2025.tif"

# --- Read raster data ---
with rasterio.open(raster_path) as src:
    arr = src.read(1)
    arr = np.nan_to_num(arr, nan=0)

    # Normalize pixel values so Folium can color them
    arr_min, arr_max = np.nanpercentile(arr, [2, 98])
    arr_norm = (arr - arr_min) / (arr_max - arr_min + 1e-9)
    arr_norm = np.clip(arr_norm, 0, 1)

    # 🌿 ArcGIS-style green symbology (YlGn)
    rgba = (cm.get_cmap("YlGn")(arr_norm) * 255).astype("uint8")

    # Convert to PNG image
    img = Image.fromarray(rgba)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    img_url = f"data:image/png;base64,{b64}"

    # Geographic bounds (south, west, north, east)
    bounds = [[src.bounds.bottom, src.bounds.left],
              [src.bounds.top, src.bounds.right]]

# --- Create simple Folium map ---
fmap = folium.Map(location=[-0.8, 37.45], zoom_start=10, tiles="CartoDB positron")

# Add raster overlay
folium.raster_layers.ImageOverlay(
    image=img_url,
    bounds=bounds,
    opacity=0.8,
    name="NDVI_2025 (green)"
).add_to(fmap)

# Layer control
folium.LayerControl().add_to(fmap)

# Save to HTML file
fmap.save("test_map.html")

print("✅ Saved test_map.html — open this file in your browser.")
