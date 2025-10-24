"""
setup_project.py
----------------
This script prepares the folder structure for the AI-Based Rice Growth & Yield Monitoring System
and organizes your data in a clean format. 

It does NOT hardcode your computer's D: path — instead, it builds a portable structure:
  AI yield system/
      ├── CSV/
      ├── rasters/
      └── shapefiles/

After running this file, copy your CSV, raster (.tif), and shapefile (.shp) data into those folders.
"""

import os

# Define folder structure
folders = [
    "CSV",
    "Raster",
    "Shapefiles",
    "outputs",
    "outputs/models",
    "outputs/artifacts"
]

def create_folders():
    print("📁 Creating project folder structure...")
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"  ✓ {folder}")
    print("\n✅ Folder setup complete!")
    print("Now copy your data files into the 'AI yield system' folder as follows:")
    print("  • CSV files (e.g., Mwea_data.csv) → AI yield system/CSV/")
    print("  • Raster files (.tif) → AI yield system/rasters/")
    print("  • Shapefiles (.shp + .shx + .dbf + .prj) → AI yield system/shapefiles/")
    print("\nOnce done, run:")
    print("  streamlit run streamlit_app.py")
    print("to start the dashboard.")

if __name__ == "__main__":
    create_folders()
