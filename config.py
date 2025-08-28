# config.py
# Global configuration for EPCI processing pipeline

# Base directory containing input data (EPCI layer, BD TOPO, OCS GE, RPG, etc.)
BASE_DIR = "/Users/anthoninkumm/Desktop/DATA_AUTOMATISATION/DATA"

# Output directory for all generated files
OUTPUT_DIR = "/Users/anthoninkumm/Desktop/DATA_AUTOMATISATION/Figure mémoire/RESULTAT_AUTOECOMAP"

# Shapefile containing all study areas (EPCI polygons)
STUDY_AREA_SHAPEFILE = "/Users/anthoninkumm/Desktop/DATA_AUTOMATISATION/DATA/ZONE_ETUDE/ZONE_PRED.shp"

# Field name in the EPCI shapefile for EPCI names
NAME_FIELD = "NOM"

# Buffer distance (in meters) to apply around each EPCI area
BUFFER_DISTANCE = 5000

# Output raster resolution (in meters per pixel)
RESOLUTION = 5

# Number of parallel workers for certain processing tasks
import os
NUM_WORKERS = max(1, int(os.cpu_count() * 0.3))  # e.g., 30% of CPU cores

# Land cover class code representing built-up areas (for final weighting)
BUILDING_CLASS_CODE = 29

# Save intermediate vector outputs (True will create per-EPCI subfolders, False will only save final rasters)
SAVE_VECTOR_OUTPUTS = True

MAX_FRICTION = 10000

