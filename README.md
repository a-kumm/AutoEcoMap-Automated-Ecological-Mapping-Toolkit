# Automated Land Cover and Friction Map Generation for Ecological Modeling

This repository provides a fully automated pipeline to generate **land cover maps (LCC)** and **friction maps** from multiple geospatial datasets (IGN BD TOPO, OCS GE, RPG, RGE ALTI, etc.).
The workflow was designed for ecological modeling, with a focus on **landscape connectivity analysis** (e.g., for *Cervus elaphus* / red deer), but can be adapted for other ecological studies.

The pipeline integrates:

* Vector preprocessing of multiple sources (land use, hydrography, vegetation, transport networks, buildings, fences, etc.).
* Automated merging of datasets across multiple departments or regions.
* Rasterization of land cover classes at a chosen resolution.
* Derivation of friction maps based on land cover, slope (DTM), and distance to built-up areas.
* Scenario-based post-processing (e.g., with/without fences or infrastructures).

---

## 📂 Project Structure

```
.
├── config.py                # Global configuration (paths, parameters, CRS, resolution, etc.)
├── data_processing_final.py # Vector preprocessing of all layers (OCS GE, BD TOPO, RPG, etc.)
├── raster_processing_final.py # Rasterization, slope weighting, building distance weighting
├── utils_ocs_final.py       # Utility functions for file handling, clipping, dataset checks
├── test_main.py             # Example script to run the pipeline on a test study area
└── README.md                # Documentation (this file)
```

---

## ⚙️ Requirements

* Python ≥ 3.9
* Recommended packages:

  ```bash
  geopandas
  shapely
  rasterio
  numpy
  pandas
  tqdm
  fiona
  pyproj
  rtree
  scikit-image
  ```

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

1. **Prepare input datasets**
   Place the required IGN datasets (BD TOPO, OCS GE, RPG, RGE ALTI) in the `base_dir` defined in `config.py`.
   The code automatically detects which departments/regions intersect the study area.

2. **Set study area**
   Define your study area in `config.py` (either by name or by providing a shapefile).

3. **Run the pipeline**
   Example with the test script:

   ```bash
   python test_main.py
   ```

   This will:

   * Clip and preprocess all layers to the study area.
   * Merge vector layers from multiple departments if necessary.
   * Generate two rasters:

     * **Land cover map** (raw classes)
     * **Friction map** (landscape permeability, with slope and building distance weighting)

4. **Output**
   Results are stored in the `OUTPUT` directory (path set in `config.py`):

   * `landcover.tif` → Land cover map (GeoTIFF, 5m resolution by default)
   * `friction.tif` → Friction map (GeoTIFF, 5m resolution by default)

---

## 🧩 Scenarios

The pipeline supports scenario-based ecological modeling by modifying friction values:

* **Full scenario** → All obstacles (fences, transport infrastructures) considered.
* **No fences** → Fence friction values neutralized.
* **No permeability** → Obstacles considered impassable.
* **Custom** → User-defined modifications.

Scenarios are applied in **post-processing** from the main friction raster.

---

## 📖 Citation

If you use this pipeline in your research, please cite it as:

> Anthonin KUMM. (2025). *Génération automatique d'occupations du sol pour modéliser l'impact des clôtures sur la connectivité écologique : application au Cerf élaphe (Cervus elaphus)*.

---

## 📝 License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

Would you like me to also create a **`requirements.txt`** file from your code (extracting the imports), so that the `README` links directly to it? That way, users can just `pip install -r requirements.txt` without guessing dependencies.
