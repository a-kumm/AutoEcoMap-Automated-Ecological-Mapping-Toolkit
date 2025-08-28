import os
import re
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
import config as cfg 
import glob
from rasterio.merge import merge
from rasterio.warp import Resampling as WarpResampling
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.mask import mask, geometry_mask
from shapely.geometry import mapping
from scipy.stats import mode

import datetime



# === SQL FILTERING UTILITY ===
def apply_SQL_filter(gdf, filter_str):
    """
    Apply a SQL-like filter string to a GeoDataFrame using pandas.query.
    Supports: =, IN, AND, OR, parentheses.
    """

    if not isinstance(filter_str, str) or filter_str.strip() == "":
        return gdf

    # Step 1: Clean up and normalize operators
    expr = filter_str.strip()
    expr = re.sub(r'\bAND\b', 'and', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bOR\b', 'or', expr, flags=re.IGNORECASE)
    expr = expr.replace("=", "==")
    expr = expr.replace("===", "==")

    # Step 2: Convert SQL-like IN (...) → Python style: .isin([...])
    def convert_in_clause(match):
        field = match.group(1)
        values = match.group(2)
        value_list = [v.strip().strip("'\"") for v in values.split(",")]
        return f"{field}.isin({value_list})"
    
    expr = re.sub(r'"(\w+)"\s+in\s+\(([^)]+)\)', convert_in_clause, expr)

    # Step 3: Convert "FIELD" to FIELD for pandas query syntax
    expr = re.sub(r'"(\w+)"', r'\1', expr)

    # Step 4: Evaluate the expression
    try:
        return gdf.query(expr)
    except Exception as e:
        print(f"⚠️ Fallback query() failed for: {expr}\n→ {e}")
        return gdf.iloc[0:0]

# === Load classification table from CSV ===
def load_table_from_csv(csv_path, vector_layers):
    import pandas as pd

    try:
        df = pd.read_csv(csv_path, sep=";", dtype=str).fillna("")
    except Exception as e:
        print(f"❌ Error reading CSV file: {csv_path}\n→ {e}")
        return pd.DataFrame()

    df.columns = [col.strip().upper().replace('\ufeff', '') for col in df.columns]

    if "SOURCE" not in df.columns:
        raise ValueError("❌ Missing 'SOURCE' column in CSV file.")

    df["SOURCE"] = df["SOURCE"].str.strip().str.upper()
    df["FILTRE"] = df.get("FILTRE", "").astype(str).fillna("").str.strip()

    required_cols = ["SOURCE", "ORDRE_COMPILATION", "COEFF_FRICTION"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Missing column: {col}")

    df["ORDRE_COMPILATION"] = pd.to_numeric(df["ORDRE_COMPILATION"], errors="coerce")
    df["COEFF_FRICTION"] = pd.to_numeric(df["COEFF_FRICTION"], errors="coerce")
    df = df.dropna(subset=["ORDRE_COMPILATION", "COEFF_FRICTION"])

    available_sources = {k.upper() for k in vector_layers.keys()}
    sources_in_csv = set(df["SOURCE"])
    missing_sources = sources_in_csv - available_sources
    if missing_sources:
        print(f"⚠️ Sources not found in vector layers: {sorted(missing_sources)}")
        df = df[~df["SOURCE"].isin(missing_sources)]

    return df.sort_values("ORDRE_COMPILATION")

# === CREATION DES RASTERS DE PAYSAGE / FRICTION 
def rasterize_classes_and_friction (
        table_df, 
        vector_layers,
        extent, 
        crs_ref,
        resolution,
        output_dir,
        area_name_clean
):
    
    # Preparation logs text 
    logs = []

    def log(msg):
        logs.append(msg)
    
    #Harmonositation des clés 
    vector_layers = {k.upper(): v for k, v in vector_layers.items()}
    table_sources = sorted(table_df['SOURCE'].dropna().unique())
    available_sources = list(vector_layers.keys())

    log("📋 Sources demandées dans la table : " + str(table_sources))
    log("🗂️ Couches vectorielles disponibles : " + str(available_sources))

    missing_sources = [s for s in table_sources if s not in available_sources ]
    if missing_sources :
        log("⚠️ Certaines sources sont absentes des couches_vectorielles : " + str(missing_sources))
    
    minx, miny, maxx, maxy = extent.bounds
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    transform = from_origin(minx, maxy, resolution, resolution)

    raster_classes = np.zeros((height, width), dtype=np.uint16)
    raster_permeability = np.zeros((height, width), dtype=np.uint16)

    classes_rasterize = set()

    for _, row in table_df.iterrows():
        sources_name = str(row["SOURCE"]).strip().upper()
        filter = str(row.get("FILTRE", "")).strip()
        if not filter or filter.lower() in ["nan", "none"]:
            filter = ""
        
        try:
            code_classes = int(row["ORDRE_COMPILATION"])
            permeability_value = int(row["COEFF_FRICTION"])
        except Exception as e : 
            log(f"⚠️ Erreur conversion numérique (classe {row}) → {e}")
            continue
        
        if sources_name not in vector_layers:
            log(f"⚠️ Couche source introuvable : {sources_name}")
            continue

        gdf = vector_layers[sources_name]
        gdf_filter = apply_SQL_filter(gdf, filter)

        if not filter : 
            log(f"📥 {sources_name} No filters → {len(gdf_filter)} objects")
        else:
            log(f"🔍 {sources_name} | filter : {filter} → {len(gdf_filter)} objects")

        if gdf_filter.empty : 
            log(f"ℹ️ No objects for {sources_name} with filter : {filter}")
        try:
            shapes = ((geom, 1) for geom in gdf_filter.geometry if geom and not geom.is_empty)
            mask = rasterize(
                shapes,
                out_shape=raster_classes.shape,
                transform=transform,
                fill=0,
                dtype = np.uint16
            )
        except Exception as e :
            log(f"❌ Erreur de rasterisation pour {sources_name} [{filter}] → {e}")
            continue

        raster_classes[mask == 1] = code_classes
        raster_permeability[mask == 1] = permeability_value
        classes_rasterize.add(code_classes)

        log(f"🔢 Add : CLASSE {code_classes} | permeability {permeability_value} | objects : {len(gdf_filter)}")
        log("")

    mask_geom = geometry_mask(
        [mapping(extent)],
        transform=transform,
        invert=True,
        out_shape=(int(raster_classes.shape[0]), int(raster_classes.shape[1]))
    )
    raster_classes[~mask_geom] = 0 
    raster_permeability[~mask_geom] = 0

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    meta = {
        "driver": "GTiff",
        "height": raster_classes.shape[0], 
        "width": raster_classes.shape[1],
        "count": 1,
        "dtype": raster_classes.dtype,
        "crs": crs_ref,
        "transform": transform
    }

    path_classes = os.path.join(output_dir, f"Raster_Classe_{area_name_clean}.tif")
    with rasterio.open(path_classes, "w", **meta) as dst:
        dst.write(raster_classes, 1)
    
    meta["dtype"] = raster_permeability.dtype
    path_permeability = os.path.join(output_dir, f"Raster_Friction_{area_name_clean}.tif")
    with rasterio.open(path_permeability, "w", **meta) as dst:
        dst.write(raster_permeability, 1)

    expected_classes = sorted(table_df["ORDRE_COMPILATION"].astype(int).unique())
    missing = [c for c in expected_classes if c not in classes_rasterize]

    # === Print Log === 
    log(f"✅ Raster classes exporté : {path_classes}")
    log(f"✅ Raster friction exporté : {path_permeability}")
    log("🚫 Classes manquantes : " + str(missing))
    log(f"\n🌟 Rasterisation : {len(classes_rasterize)}/{len(expected_classes)} classes présentes ({100 * len(classes_rasterize) / len(expected_classes):.1f} %)")

    # == Final log === 
    log_path = os.path.join(cfg.OUTPUT_DIR, f"Rasterization_log_{area_name_clean}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"🗓️ Log generate the : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"EPCI process : {area_name_clean}\n")
        f.write("="*60 + "\n\n")
        f.write("\n".join(logs))
        f.write("\n\n" + "="*60 + "\nEnd of log.\n")
    
    log(f"\n Log exported to : {log_path}")
   
# === DEM Processing and Friction Weighting ===
def process_dtm_from_tiles(geom_extent, crs_ref, base_dir, output_mnt_path):
    """
    Merges and crops RGE ALTI tiles from multiple departments intersecting the study area.
    """

    # Étape 1 — Charger tous les DEPARTEMENT.shp trouvés
    dep_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(base_dir)
        for file in files
        if file.lower() == "departement.shp"
    ]
    if not dep_files:
        raise FileNotFoundError("❌ No DEPARTEMENT.shp file found in base_dir.")

    # Fusionner tous les départements trouvés
    gdfs = []
    for path in dep_files:
        gdf = gpd.read_file(path)
        if gdf.crs != crs_ref:
            gdf = gdf.to_crs(crs_ref)
        gdfs.append(gdf)
    gdf_dep = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=crs_ref)

    # Extraire les départements intersectés
    gdf_clip = gdf_dep[gdf_dep.geometry.intersects(geom_extent)]
    if gdf_clip.empty:
        raise RuntimeError("❌ No departments intersect with the buffer geometry.")

    code_field = next((col for col in gdf_clip.columns if "DEP" in col.upper() or "CODE" in col.upper()), None)
    code_depts = gdf_clip[code_field].astype(str).str.zfill(3).unique().tolist()

    # Étape 2 — Trouver les tuiles RGE ALTI pour ces départements
    tiles_path = []
    for code in code_depts:
        rge_dirs = [
            os.path.join(base_dir, d) for d in os.listdir(base_dir)
            if d.startswith("RGEALTI_") and f"D{code}" in d
        ]
        if not rge_dirs:
            print(f"⚠️ No RGEALTI folder found for department D{code}")
            continue

        for dir_path in rge_dirs:
            tiles_shp = glob.glob(os.path.join(dir_path, "**", "dalles.shp"), recursive=True)
            if not tiles_shp:
                print(f"⚠️ No dalles.shp found in {dir_path}")
                continue

            gdf_tiles = gpd.read_file(tiles_shp[0])
            if gdf_tiles.crs != crs_ref:
                gdf_tiles = gdf_tiles.to_crs(crs_ref)

            gdf_selection = gdf_tiles[gdf_tiles.intersects(geom_extent)]

            for _, row in gdf_selection.iterrows():
                tile_name = row["NOM_DALLE"]
                for ext in [".asc", ".tif"]:
                    match = glob.glob(os.path.join(dir_path, "**", tile_name + ext), recursive=True)
                    if match:
                        tiles_path.append(match[0])
                        break

    if not tiles_path:
        raise FileNotFoundError("❌ No RGE ALTI tiles found for the selected departments.")

    # Fusion et crop des tuiles
    src_files = []
    for fp in tiles_path:
        src = rasterio.open(fp)
        if src.crs is None:
            profile = src.profile
            profile.update(crs=crs_ref)
            data = src.read()
            memfile = rasterio.io.MemoryFile()
            tmp = memfile.open(**profile)
            tmp.write(data)
            src_files.append(tmp)
        else:
            src_files.append(src)

    mosaic, out_trans = merge(src_files)
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": crs_ref
    })

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**out_meta) as tmp:
            tmp.write(mosaic)
            out_image, out_transform = mask(tmp, [mapping(geom_extent)], crop=True)

    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    with rasterio.open(output_mnt_path, "w", **out_meta) as dst:
        dst.write(out_image)

  
# === SLOPE WEIGHTING
def apply_slope_weighting(
    path_permeability,
    path_dtm,
    path_output,
    max_friction=10000
):
    """
    Applies slope-based weighting to a friction raster.

    - 0 ≤ slope < 30°   → ×1
    - 30 ≤ slope < 40°  → ×10
    - slope ≥ 40°       → ×1000

    Args:
        path_permeability (str): Path to the input friction raster.
        path_dtm (str): Path to the input DTM raster (elevation).
        path_output (str): Path to save the weighted friction raster.
        max_friction (int): Maximum allowed friction value (default: 10000).

    """
    import numpy as np
    import rasterio
    from rasterio.warp import reproject, Resampling as WarpResampling
    from scipy.ndimage import sobel

    # Load friction raster
    with rasterio.open(path_permeability) as src_friction:
        friction = src_friction.read(1)
        profile = src_friction.profile
        ref_transform = src_friction.transform
        ref_crs = src_friction.crs
        ref_shape = friction.shape

    # Reproject DTM to friction raster grid
    with rasterio.open(path_dtm) as src_mnt:
        dtm = src_mnt.read(1)
        dtm_transform = src_mnt.transform
        dtm_crs = src_mnt.crs
        dtm_reproj = np.empty(ref_shape, dtype=np.float32)

        reproject(
            source=dtm,
            destination=dtm_reproj,
            src_transform=dtm_transform,
            src_crs=dtm_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=WarpResampling.bilinear
        )

    # Compute slope in degree
    pixel_size = ref_transform[0]
    dzdx = sobel(dtm_reproj, axis=1) / (8 * pixel_size)
    dzdy = sobel(dtm_reproj, axis=0) / (8 * pixel_size)
    slope_degree = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

    # Create weighting matrix
    weighting_factor = np.ones_like(friction, dtype=np.float32)
    weighting_factor[(slope_degree >= 30) & (slope_degree < 40)] = 10
    weighting_factor[slope_degree >= 40] = 1000

    # Apply slope-based weighting
    weighting_friction = np.round(friction * weighting_factor).astype(np.uint16)
    weighting_friction = np.clip(weighting_friction, 0, max_friction)

    # Save output raster
    profile.update(dtype=rasterio.uint16)
    with rasterio.open(path_output, "w", **profile) as dst:
        dst.write(weighting_friction, 1)

# === BUILT-UP AREA WEIGHTING
def apply_building_distance_weighting(
    path_permeability,
    path_raster_classes,
    building_class_code,
    path_output
):
    """
    Applies a multiplicative weighting to a friction raster based on distance to built-up areas.

    Thresholds:
        - distance ≤ 50 m     → ×2.5
        - 50 < distance ≤ 100 m → ×2
        - 100 < distance ≤ 200 m → ×1.5
        - distance > 200 m    → ×1 (no change)

    """
    import numpy as np
    import rasterio
    from scipy.ndimage import distance_transform_edt

    # 1. Load friction raster
    with rasterio.open(path_permeability) as src_friction:
        friction = src_friction.read(1).astype(np.float32)
        profile = src_friction.profile
        transform = src_friction.transform
        resolution = transform[0]

    # 2. Load land cover class raster
    with rasterio.open(path_raster_classes) as src_classes:
        classes = src_classes.read(1)

    # 3. Create binary mask of built-up area
    building_mask = (classes == building_class_code)

    # 4. Calcul distance en pixels
    distance_pixels = distance_transform_edt(~building_mask)

    # 5. Convert distance to meters
    distance_m = distance_pixels * resolution

    # 6. Create weighting factor matrix 
    weighting_factor = np.ones_like(friction, dtype=np.float32)
    weighting_factor[distance_m <= 50] = 2.5
    weighting_factor[(distance_m > 50) & (distance_m <= 100)] = 2.0
    weighting_factor[(distance_m > 100) & (distance_m <= 200)] = 1.5
    # >200m → 1 (déjà par default)

    # 7. Apply the weighting 
    weighting_friction = np.round(friction * weighting_factor).astype(np.uint16)
    weighting_friction = np.clip(weighting_friction, 0, 10000)

    # 8. Export result
    profile.update(dtype=rasterio.uint16)
    with rasterio.open(path_output, "w", **profile) as dst:
        dst.write(weighting_friction, 1)

    print(f"✅ Friction raster weighted by building proximity saved to : {path_output}")

# === INTERPOLATION DE LA PERMEABILITE POUR LES DIFFERENTS SCENARIO === 
def replace_obstacle_friction_by_local_interp(
    path_friction,
    path_classes,
    target_class_codes,
    path_output,
    nodata_value=0,
    window_size=10
):
    """
    Optimisé : remplace uniquement les pixels de friction associés à des obstacles à supprimer (clôtures, ILT)
    par la valeur la plus fréquente parmi les voisins valides.

    Args:
        path_friction (str): Chemin du raster de friction.
        path_classes (str): Chemin du raster de classes.
        target_class_codes (list): Codes de classe à neutraliser (ex: [998, 999]).
        path_output (str): Chemin du fichier de sortie.
        nodata_value (int or float): Valeur nodata à ignorer.
        window_size (int): Taille de la fenêtre de voisinage (impair, ex: 3 → 3x3).
    """
    print(f"\n🔄 Interpolation optimisée des frictions pour suppression des classes {target_class_codes}...")

    with rasterio.open(path_friction) as src_friction, rasterio.open(path_classes) as src_classes:
        friction = src_friction.read(1).astype(np.float32)
        classes = src_classes.read(1)
        profile = src_friction.profile

    mask_target = np.isin(classes, target_class_codes)
    h, w = friction.shape
    pad = window_size // 2

    # Padding de l'image de friction
    padded_friction = np.pad(friction, pad_width=pad, mode="reflect")
    result = friction.copy()

    # Liste des pixels à corriger
    target_indices = np.argwhere(mask_target)

    for idx, (i, j) in enumerate(target_indices):
        i_p, j_p = i + pad, j + pad
        window = padded_friction[i_p - pad:i_p + pad + 1, j_p - pad:j_p + pad + 1].flatten()

        # Exclure le centre + les nodata ou NaN
        neighbors = np.delete(window, len(window) // 2)
        neighbors = neighbors[
            (neighbors != nodata_value) &
            (neighbors != 10000) &  # <- On exclut les pixels très résistants
            (~np.isnan(neighbors))
        ]

        if len(neighbors) > 0:
            most_common = mode(neighbors, keepdims=False).mode
            if most_common.size > 0:
                result[i, j] = most_common.item()

        if idx % 10000 == 0 and idx > 0:
            print(f"  → {idx}/{len(target_indices)} pixels traités...")

    # Écriture du résultat
    profile.update(dtype=rasterio.float32)
    with rasterio.open(path_output, "w", **profile) as dst:
        dst.write(result, 1)

    print(f"✅ Friction corrigée enregistrée dans : {path_output}")

# === SUPPRESSION DU BIAI === 
def replace_class3_1000_by_local_mode(
    path_friction: str,
    path_classes: str,
    output_path: str,
    window_size: int = 11,        # 10 -> 11x11 comme ta logique
    nodata_value: float | int = 0
):
    """
    PARTOUT : remplace UNIQUEMENT les pixels où (classe == 3) ET (friction == 1000)
    par la valeur la plus fréquente (mode) des voisins dans une fenêtre glissante,
    en excluant seulement nodata et NaN des voisins. 👉 10 000 est accepté comme donneur.
    """
    import numpy as np
    import rasterio

    with rasterio.open(path_friction) as src_f, rasterio.open(path_classes) as src_c:
        friction = src_f.read(1).astype(np.float32)
        classes = src_c.read(1)
        profile = src_f.profile

    # Cibles : classe 3 & friction 1000 (partout)
    mask_target = (classes == 3) & (friction == 1000)
    if not np.any(mask_target):
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(friction, 1)
        print(f"ℹ️ Aucun pixel (classe 3 & friction=1000) → copie : {output_path}")
        return output_path

    h, w = friction.shape
    pad = window_size // 2

    # Padding reflect
    padded_friction = np.pad(friction, pad_width=pad, mode="reflect")
    result = friction.copy()

    target_indices = np.argwhere(mask_target)
    for idx, (i, j) in enumerate(target_indices):
        i_p, j_p = i + pad, j + pad
        window = padded_friction[i_p - pad:i_p + pad + 1, j_p - pad:j_p + pad + 1].flatten()

        # Exclure le centre + nodata + NaN (⚠️ 10 000 autorisé comme voisin)
        neighbors = np.delete(window, len(window) // 2)
        neighbors = neighbors[
            (neighbors != nodata_value) &
            (~np.isnan(neighbors))
        ]

        if neighbors.size > 0:
            m = mode(neighbors, keepdims=False).mode
            if m.size > 0:
                result[i, j] = m.item()

        if idx % 10000 == 0 and idx > 0:
            print(f"  → {idx}/{len(target_indices)} pixels traités...")

    profile.update(dtype=rasterio.float32)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(result, 1)

    print(f"✅ Correction globale (classe=3 & friction=1000) par mode → {output_path}")
    return output_path
