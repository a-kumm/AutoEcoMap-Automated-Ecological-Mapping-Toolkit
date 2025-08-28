import os
import pandas as pd
import geopandas as gpd
import config as cfg
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from utils_ocs_final import find_file,load_gdf
from pathlib import Path

"""
________________________________________________________________________________
____

<Module Description and Usage>

________________________________________________________________________________
____
"""



# === CPU LOAD LIMITATION ===
num_cores = os.cpu_count()
num_workers = max(1, int(num_cores * 0.3))  # Limit to ~30% of CPU cores

# === Land Cover Process (OCS GE) ===
def process_land_cover_data(base_dir, output_dir, extent, area_name_clean, num_workers):
    """
    Loads and clips all OCS GE (land cover) files to the provided extent, then merges them.
    """

    # 1. Retrieve all land cover files 
    land_cover_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(base_dir)
        for file in files
        if "OCCUPATION_SOL" in file.upper() and file.lower().endswith((".shp", ".gpkg"))
    ]

    if not land_cover_files:
        raise RuntimeError("❌ No OCCUPATION_SOL file found in base_dir")

    # 2. Parallel reading and clipping 
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        gdfs_ocs = list(executor.map(
            partial(
                load_gdf,
                useful_columns=["ID", "CODE_CS", "CODE_US", "GEOMETRY"],
                force_id=True,
                clip_geom=extent
            ),
            land_cover_files
        ))

    # 3. Merge land cover (OCS GE)
    gdf_ocs_clip = gpd.GeoDataFrame(
        pd.concat(gdfs_ocs, ignore_index=True),
        geometry="GEOMETRY",
        crs=gdfs_ocs[0].crs
    )

    # 4. Export merged clip
    export_clip = os.path.join(output_dir, f"OCSGE_{area_name_clean}.shp")
    if cfg.SAVE_VECTOR_OUTPUTS:
        gdf_ocs_clip.to_file(export_clip)

    return gdf_ocs_clip

# === RPG Parcels Process ===
def process_rpg_data(base_dir, output_dir, extent, area_name_clean, crs_ref):
    """
    Loads and clips RPG parcels from all departments intersecting the study area.
    Only files matching department codes in their path are processed.
    """

    #1. Identify intersected regions 
    region_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower() == "region.shp":
                region_files.append(os.path.join(root, f))
    if not region_files:
        raise FileNotFoundError ("❌ No region files found for RPG data")
    
    gdfs = []
    for fp in region_files:
        g = gpd.read_file(fp)
        if g.crs is None:
            g.set_crs(crs_ref, inplace=True)
        elif g.crs != crs_ref:
            g = g.to_crs(crs_ref)
        gdfs.append(g)
    gdf_regions = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), geometry="geometry", crs=crs_ref)

    extent_geom = extent if hasattr(extent, "geom_type") else extent.unary_union
    intersecting_regions = gdf_regions[gdf_regions.geometry.intersects(extent_geom)]
    if intersecting_regions.empty:
        raise RuntimeError("❌ No regions intersect with the buffer geometry.")
    code_field = "INSEE_REG" if "INSEE_REG" in intersecting_regions.columns else next(c for c in intersecting_regions.columns if "REG" in c.upper())
    code_region = intersecting_regions[code_field].astype(str).unique().tolist()

    # 2. Find RPG Files 

    rpg_files = []
    for root, _, files in os.walk(base_dir):
        for f in files : 
            if "PARCELLES_GRAPHIQUES" in f.upper() and f.lower().endswith(".shp"):
                full_path = os.path.join(root, f)
                full_path_parts = Path(full_path).parts
                region_in_path = any(code in part for part in full_path_parts for code in code_region)
                if region_in_path:
                    rpg_files.append(full_path)
    
    if not rpg_files:
        raise RuntimeError("❌ No RPG files found for intersecting regions")

    # 3. Load and clipping files 

    list_rpg_gdf = []
    for path in rpg_files:
        try:
            gdf = gpd.read_file(path)
            if gdf.crs != crs_ref:
                gdf = gdf.to_crs(crs_ref)
            gdf = gdf[gdf.geometry.is_valid]
            gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
            gdf_clip = gdf[gdf.intersects(extent)].copy()
            gdf_clip["geometry"] = gdf_clip.geometry.intersection(extent)
            gdf_clip = gdf_clip[~gdf_clip.is_empty & gdf_clip.geometry.notnull()]
            list_rpg_gdf.append(gdf_clip)
        except Exception as e:
            print(f"⚠️ Erreur lors du traitement de {path} : {e}")

    if not list_rpg_gdf:
        raise RuntimeError("❌ Aucun RPG valide n'a pu être traité.")

    # 4. Fusion finale
    gdf_rpg_clip = gpd.GeoDataFrame(
        pd.concat(list_rpg_gdf, ignore_index=True),
        geometry="geometry",
        crs=crs_ref
    )

    # 5. Export
    export_rpg = os.path.join(output_dir, f"RPG_{area_name_clean}.shp")
    if cfg.SAVE_VECTOR_OUTPUTS:
        gdf_rpg_clip.to_file(export_rpg)

    return gdf_rpg_clip

# === Vegetation & Hedgerows (BD TOPO) ===
def process_vegetation_data(base_dir, output_dir, extent, area_name_clean, crs_ref):
    """
    Processes vegetation areas and hedgerows from BD TOPO (multi-department):
    - Merges and clips HAIE (hedge) layers and ZONE_DE_VEGETATION (filtered for hedges) layers.
    - Applies a 2.5 m buffer to hedge lines and merges with vegetation polygons.
    """
    from pathlib import Path

    # 1. Identifier tous les départements intersectant l'étendue
    dept_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower() == "departement.shp":
                dept_files.append(os.path.join(root, f))
    if not dept_files:
        raise FileNotFoundError("❌ Aucune couche DEPARTEMENT trouvée pour la végétation")
    gdfs = []
    for fp in dept_files:
        g = gpd.read_file(fp)
        if g.crs is None:
            g.set_crs(crs_ref, inplace=True)
        elif g.crs != crs_ref:
            g = g.to_crs(crs_ref)
        gdfs.append(g)
    gdf_departments = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), geometry="geometry", crs=crs_ref)

    extent_geom = extent if hasattr(extent, "geom_type") else extent.unary_union
    intersecting_deps = gdf_departments[gdf_departments.geometry.intersects(extent_geom)]
    if intersecting_deps.empty:
        raise RuntimeError("❌ No departments intersect with the buffer geometry.")
    code_field = "INSEE_DEP" if "INSEE_DEP" in intersecting_deps.columns else next(c for c in intersecting_deps.columns if "DEP" in c.upper())
    codes_dep = intersecting_deps[code_field].astype(str).unique().tolist()

    # 2. Recherche des fichiers végétation et haie pour ces départements (dans tout le chemin)
    vege_files, hedge_files = [], []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith((".shp", ".gpkg")):
                full_path = os.path.join(root, f)
                full_path_parts = Path(full_path).parts
                dep_in_path = any(code in part for part in full_path_parts for code in codes_dep)

                if "ZONE_DE_VEGETATION" in f.upper() and dep_in_path:
                    vege_files.append(full_path)
                elif f.upper().startswith("HAIE") and dep_in_path:
                    hedge_files.append(full_path)

    if not vege_files:
        raise RuntimeError("❌ No vegetation files found.")
    if not hedge_files:
        raise RuntimeError("❌ No hedge files found.")

    # 3. Clip vegetation areas
    list_vege_gdf = [
        load_gdf(path, useful_columns=["ID", "NATURE", "GEOMETRY"], clip_geom=extent)
        for path in vege_files
    ]
    gdf_vege_clipped = gpd.GeoDataFrame(
        pd.concat(list_vege_gdf, ignore_index=True).rename(columns={"GEOMETRY": "geometry"}),
        geometry="geometry",
        crs=crs_ref
    )

    # Clean the 'NATURE' field 
    if "NATURE" in gdf_vege_clipped.columns:
        import unicodedata
        gdf_vege_clipped["NATURE"] = (
            gdf_vege_clipped["NATURE"]
            .fillna("")
            .apply(lambda x: unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode('utf-8').strip().replace(" ", "_").upper())
        )

    export_vege = os.path.join(output_dir, f"Vegetation_{area_name_clean}.shp")
    if cfg.SAVE_VECTOR_OUTPUTS:
        gdf_vege_clipped.to_file(export_vege)

    # 4. Extract hedges from vegetation layer (NATURE == 'HAIE')
    gdf_hedges_from_vege = gdf_vege_clipped[gdf_vege_clipped.get("NATURE", "") == "HAIE"][["geometry"]].copy()
    gdf_vege_clipped = gdf_vege_clipped[gdf_vege_clipped.get("NATURE", "") != "HAIE"].copy()

    # 5. Clip HAIE layers and apply 2.5 m buffer
    hedge_segments = []
    for path in hedge_files:
        gdf_h = gpd.read_file(path)
        if gdf_h.crs != crs_ref:
            gdf_h = gdf_h.to_crs(crs_ref)
        gdf_h_clip = gpd.clip(gdf_h, extent)
        gdf_h_clip["geometry"] = gdf_h_clip.geometry.buffer(2.5)
        hedge_segments.append(gdf_h_clip[["geometry"]])

    # 6. Merge hedge sources (from vegetation zones and HAIE layers)
    gdf_hedges_all = gpd.GeoDataFrame(
        pd.concat(hedge_segments + [gdf_hedges_from_vege], ignore_index=True),
        geometry="geometry",
        crs=crs_ref
    )

    # 7. Dissolve hedge geometries
    merged_hedges = gdf_hedges_all.geometry.unary_union
    gdf_hedges_merge = gpd.GeoDataFrame(
        geometry=[geom for geom in merged_hedges.geoms] if hasattr(merged_hedges, "geoms") else [merged_hedges],
        crs=crs_ref
    ).reset_index(drop=True)
    gdf_hedges_merge["ID"] = range(1, len(gdf_hedges_merge) + 1)

    # 8. Export results
    export_hedges = os.path.join(output_dir, f"Hedges_{area_name_clean}.shp")
    if cfg.SAVE_VECTOR_OUTPUTS:
        gdf_hedges_merge.to_file(export_hedges)

    return gdf_vege_clipped, gdf_hedges_merge

# === Hydrographic Network (BD TOPO) ===
def process_hydrography_network(base_dir, output_dir, extent, area_named_clean, crs_ref):
    """
    Processes the BD TOPO hydrographic network (Surface + Segment) across multiple departments.
    Applies clipping, buffering, stream separation, overlap removal, and final merging.
    Transfers the WIDTH from segments to surfaces where possible.
    """

    import geopandas as gpd
    import pandas as pd
    import os


    debug_dir = os.path.join(output_dir, "hydro_processing")
    if cfg.SAVE_VECTOR_OUTPUTS:
        os.makedirs(debug_dir, exist_ok=True)

    # 1. Identifier tous les départements intersectant l'étendue
    dept_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower() == "departement.shp":
                dept_files.append(os.path.join(root, f))
    if not dept_files:
        raise FileNotFoundError("❌ Aucune couche DEPARTEMENT trouvée pour le réseau hydrographique")
    gdfs = []
    for fp in dept_files:
        g = gpd.read_file(fp)
        if g.crs != crs_ref:
            g = g.to_crs(crs_ref)
        gdfs.append(g)
    gdf_departments = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), geometry="geometry", crs=crs_ref)

    intersecting_deps = gdf_departments[gdf_departments.geometry.intersects(extent)]
    code_field = "INSEE_DEP" if "INSEE_DEP" in intersecting_deps.columns else next(c for c in intersecting_deps.columns if "DEP" in c.upper())
    codes_dep = intersecting_deps[code_field].astype(str).unique().tolist()

    # === Search files 
    surface_files, hydro_sections_files = [], []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith((".shp", ".gpkg")):
                path = os.path.join(root, f)
                if "SURFACE_HYDROGRAPHIQUE" in f.upper() and any(code in path for code in codes_dep):
                    surface_files.append(path)
                elif "TRONCON_HYDROGRAPHIQUE" in f.upper() and any(code in path for code in codes_dep):
                    hydro_sections_files.append(path)

    if not surface_files or not hydro_sections_files:
        raise RuntimeError("❌ Missing Hydrographic data")

    columns_surface = ['ID', 'CODE_HYDRO', 'NATURE', 'POS_SOL', 'ETAT', 'PERSISTANC', 'SALINITE', 'ORIGINE', 'geometry']
    columns_hydro_sections = ['ID', 'CODE_HYDRO', 'NATURE', 'FICTIF', 'ETAT', 'POS_SOL', 'PERSISTANC', 'LARGEUR', 'NOM_C_EAU', 'geometry']

    gdfs_surface, gdfs_hydro_sections = [], []

    for path in surface_files:
        gdf = gpd.read_file(path)
        if gdf.crs != crs_ref:
            gdf = gdf.to_crs(crs_ref)
        gdf = gdf[[col for col in columns_surface if col in gdf.columns]]
        gdf = gpd.clip(gdf, extent)
        gdf['POS_SOL'] = pd.to_numeric(gdf['POS_SOL'], errors='coerce')
        gdf = gdf[gdf['POS_SOL'] >= 0]
        gdfs_surface.append(gdf)

    for path in hydro_sections_files:
        gdf = gpd.read_file(path)
        if gdf.crs != crs_ref:
            gdf = gdf.to_crs(crs_ref)
        # Ensures the presence of watercourse widths
        valid_columns = [col for col in columns_hydro_sections if col in gdf.columns]
        gdf = gdf[valid_columns]
        gdf = gpd.clip(gdf, extent)
        gdf['POS_SOL'] = pd.to_numeric(gdf['POS_SOL'], errors='coerce')
        gdf['FICTIF'] = gdf['FICTIF'].astype(str).str.upper()
        gdf = gdf[(gdf['POS_SOL'] >= 0) & (gdf['FICTIF'] != 'OUI')]
        gdfs_hydro_sections.append(gdf)

    gdf_surface = gpd.GeoDataFrame(pd.concat(gdfs_surface, ignore_index=True), crs=crs_ref)
    gdf_section = gpd.GeoDataFrame(pd.concat(gdfs_hydro_sections, ignore_index=True), crs=crs_ref)

    if cfg.SAVE_VECTOR_OUTPUTS:
        gdf_surface.to_file(os.path.join(debug_dir, "Surface_Hydrographique_Filtre.shp"))
        gdf_section.to_file(os.path.join(debug_dir, "Troncon_Hydrographique_Filtre.shp"))

    # === Buffer and separate permanent/intermittent rivers
    gdf_buffer = gdf_section.copy()
    gdf_buffer["geometry"] = gdf_buffer.geometry.buffer(2.5)
    gdf_buffer["PERSISTANC"] = gdf_buffer["PERSISTANC"].astype(str).str.upper()
    gdf_perm = gdf_buffer[gdf_buffer["PERSISTANC"] == "PERMANENT"]
    gdf_interm = gdf_buffer[gdf_buffer["PERSISTANC"] == "INTERMITTENT"]
    gdf_interm["geometry"] = gdf_interm.geometry.difference(gdf_perm.unary_union)

    gdf_hydro_section_clean = gpd.GeoDataFrame(pd.concat([gdf_perm, gdf_interm], ignore_index=True), geometry="geometry", crs=crs_ref)
   
    if cfg.SAVE_VECTOR_OUTPUTS:
        gdf_hydro_section_clean.to_file(os.path.join(debug_dir, "Hydro_section_Merge_Buffer2m.shp"))

    # === Remove overlapping segments with surfaces

    gdf_section_no_surface = gdf_hydro_section_clean.copy()
    gdf_section_no_surface["geometry"] = gdf_section_no_surface.geometry.difference(gdf_surface.unary_union)

    if cfg.SAVE_VECTOR_OUTPUTS:
        gdf_section_no_surface.to_file(os.path.join(debug_dir, "Section_without_Surface.shp"))

    # === Transferring width informations from segments to surfaces 
    if "LARGEUR" in gdf_section_no_surface.columns:
        gdf_surface["LARGEUR"] = pd.NA
        gdf_match = gpd.sjoin(
            gdf_surface[["geometry"]].copy(),
            gdf_section_no_surface[["geometry", "LARGEUR"]].dropna(),
            how="left",
            predicate="intersects"
        )
        gdf_surface.loc[gdf_match.index, "LARGEUR"] = gdf_match.groupby(gdf_match.index)["LARGEUR"].transform("max")
    else:
        print("ℹ️ No 'LARGEUR' field found in segments, no value transferred to surfaces")

    # === Final merge
    gdf_surface["SOURCE"] = "surface"
    gdf_section_no_surface["SOURCE"] = "troncon"

    common_columns = list(set(gdf_surface.columns) & set(gdf_section_no_surface.columns))
    if "geometry" not in common_columns:
        common_columns.append("geometry")

    gdf_hydro = gpd.GeoDataFrame(
        pd.concat([
            gdf_surface[common_columns],
            gdf_section_no_surface[common_columns]
        ], ignore_index=True),
        geometry="geometry",
        crs=crs_ref
    )

    output_hydro_path = os.path.join(output_dir, f"Hydro_{area_named_clean}.shp")
    if cfg.SAVE_VECTOR_OUTPUTS:
        gdf_hydro.to_file(output_hydro_path)

    return gdf_hydro

# === Special Technical Infrastructures (BD TOPO) ===
def process_technical_infrastructure(base_dir, output_dir, extent, area_name_clean, crs_ref):
    """
    Processes special buildings/infrastructures from BD TOPO:
    CIMETIERE, RESERVOIR, TERRAIN_DE_SPORT, POSTE_DE_TRANSFORMATION, AERODROME.
    Includes spatial clipping, specific filters, and export with essential attributes.
    """

    # Columns to keep per layer 
    columns_by_type = {
        "CIMETIERE": ["ID", "NATURE", "NAT_DETAIL", "ETAT", "GEOMETRY"],
        "RESERVOIR": ["ID", "NATURE", "ETAT", "HAUTEUR", "GEOMETRY"],
        "TERRAIN_DE_SPORT": ["ID", "ETAT", "NAT_DETAIL", "NATURE", "GEOMETRY"],
        "POSTE_DE_TRANSFORMATION": ["ID", "TOPONYME", "ETAT", "GEOMETRY"],
        "AERODROME": ["ID", "CATEGORIE", "NATURE", "USAGE", "FICTIF", "ETAT", "GEOMETRY"]
    }

    layers = list(columns_by_type.keys())
    gdfs = []

    # Identify intersecting departments
    dept_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(base_dir)
        for f in files
        if f.lower() == "departement.shp"
    ]
    if not dept_files:
        raise FileNotFoundError("❌ Aucune couche DEPARTEMENT trouvée pour les bâtiments spéciaux")

    dept_gdfs = []
    for fp in dept_files:
        g = gpd.read_file(fp)
        if g.crs != crs_ref:
            g = g.to_crs(crs_ref)
        dept_gdfs.append(g)
    gdf_departments = gpd.GeoDataFrame(pd.concat(dept_gdfs, ignore_index=True), geometry="geometry", crs=crs_ref)

    # Étendue unifiée si GeoDataFrame
    extent_geom = extent.unary_union if isinstance(extent, gpd.GeoDataFrame) else extent

    intersecting_deps = gdf_departments[gdf_departments.geometry.intersects(extent_geom)]
    if intersecting_deps.empty:
        raise RuntimeError("❌ No departments intersect with the buffer geometry.")
    code_field = "INSEE_DEP" if "INSEE_DEP" in intersecting_deps.columns else next(
        c for c in intersecting_deps.columns if "DEP" in c.upper())
    codes_dep = intersecting_deps[code_field].astype(str).unique().tolist()

    # Process each layer
    for layer in layers:
        files = [
            os.path.join(root, f)
            for root, _, fs in os.walk(base_dir)
            for f in fs
            if f.upper().endswith(".SHP")
            and layer in f.upper()
            and any(code in os.path.join(root, f) for code in codes_dep)
        ]

        for path in files:
            gdf = load_gdf(
                path,
                useful_columns=columns_by_type[layer],
                force_id=True,
                clip_geom=extent_geom
            )

            if gdf.empty:
                continue

            # AERODROME filter
            if layer == "AERODROME":
                for col in ["ETAT", "NATURE", "USAGE", "FICTIF"]:
                    if col in gdf.columns:
                        gdf[col] = gdf[col].astype(str).str.strip().str.lower()
                gdf = gdf[
                    (gdf.get("ETAT") == "en service") &
                    (gdf.get("NATURE") == "aérodrome") &
                    (gdf.get("USAGE") == "civil") &
                    (gdf.get("FICTIF") == "non")
                ]
            else:
                if "ETAT" in gdf.columns:
                    gdf["ETAT"] = gdf["ETAT"].astype(str).str.strip().str.upper()
                    gdf = gdf[gdf["ETAT"] == "EN SERVICE"]

            if gdf.empty:
                continue

            gdf["TYPE"] = layer
            gdfs.append(gdf)

    # Final merge
    if gdfs:
        gdf_infrastructures = pd.concat(gdfs, ignore_index=True)

        # Convert 'GEOMETRY' to 'geometry'
        if "GEOMETRY" in gdf_infrastructures.columns:
            gdf_infrastructures = gdf_infrastructures.rename(columns={"GEOMETRY": "geometry"})

        # Drop other geometry columns if any
        geometry_columns = [
            col for col in gdf_infrastructures.columns
            if isinstance(gdf_infrastructures[col], gpd.GeoSeries) and col != "geometry"
        ]
        gdf_infrastructures = gdf_infrastructures.drop(columns=geometry_columns)

        # Ensure correct GeoDataFrame structure
        gdf_infrastructures = gpd.GeoDataFrame(gdf_infrastructures, geometry="geometry", crs=crs_ref)
    else:
        gdf_infrastructures = gpd.GeoDataFrame(columns=["geometry", "TYPE"], geometry="geometry", crs=crs_ref)

    # Export
    export_path = os.path.join(output_dir, f"Technical_infrastructures_{area_name_clean}.shp")
    if cfg.SAVE_VECTOR_OUTPUTS:
        gdf_infrastructures.to_file(export_path)

    return gdf_infrastructures

# === Wildlife crossing process (BD ORFeH)
def process_wildlife_crossing(base_dir, output_dir, extent, crs_ref, area_name_clean):
    """
    Loads the ORFeH wildlife crossing layer, clips it to the extent,
    applies a set of filters (passability, type, dimensions),
    creates a 10m buffer around valid points, and exports results.
    Returns both the clipped and buffered GeoDataFrames with all needed attributes.
    """

    path = find_file(base_dir, "ORFeH_NATIONAL_fusione_V2")
    gdf = gpd.read_file(path)
    if gdf.crs != crs_ref:
        gdf = gdf.to_crs(crs_ref)

    gdf_ORFeH_clip = gpd.clip(gdf, extent).copy()

    # Convert and clean key numeric fields
    gdf_ORFeH_clip["OA_Larg_p"] = pd.to_numeric(gdf_ORFeH_clip.get("OA_Larg_p"), errors="coerce").round(3)
    gdf_ORFeH_clip["OA_Long_p"] = pd.to_numeric(gdf_ORFeH_clip.get("OA_Long_p"), errors="coerce").round(3)

    useful_columns = [
        "Info_ID", "ILT_Type", "ILT_Nom", "ILT_Grill", "ILT_Elemen",
        "OA_Type_p", "OA_Franc_p", "OA_Larg_p", "OA_Long_p",
        "Hydro_p", "Fric_Ong", "Franch_Ong", "geometry"
    ]
    gdf_ORFeH_clip = gdf_ORFeH_clip[[col for col in useful_columns if col in gdf_ORFeH_clip.columns]].copy()

    # Apply filters: only passages that are passable and not tunnels
    gdf_ORFeH_clip = gdf_ORFeH_clip[
        (gdf_ORFeH_clip["Franch_Ong"].str.upper() == "FRANCHISSABLE") &
        (gdf_ORFeH_clip["OA_Type_p"].str.upper() != "TUNNEL OU TRANCHÉE COUVERTE")
    ]

    def valid_crossing(row):
        passage_type = str(row.get("OA_Franc_p", "")).strip().lower()
        width = row.get("OA_Larg_p")
        length = row.get("OA_Long_p")
        if passage_type == "passage sous ilt":
            return (width or 0) >= 20 and (pd.isna(length) or length <= 32)
        elif passage_type == "passage sur ilt":
            return (width or 0) >= 12
        return True

    gdf_ORFeH_clip = gdf_ORFeH_clip[gdf_ORFeH_clip.apply(valid_crossing, axis=1)].copy()

    output_raw = os.path.join(output_dir, f"ORFeH_{area_name_clean}.shp")
    if cfg.SAVE_VECTOR_OUTPUTS:
        gdf_ORFeH_clip.to_file(output_raw)

    # Buffer the geometries by 10 meters
    gdf_ORFeH_buffered = gdf_ORFeH_clip.copy()
    gdf_ORFeH_buffered["geometry"] = gdf_ORFeH_buffered.geometry.buffer(10)

    output_buffer = os.path.join(output_dir, f"ORFeH_{area_name_clean}_buffer10m.shp")
    if cfg.SAVE_VECTOR_OUTPUTS:
        gdf_ORFeH_buffered.to_file(output_buffer)

    return gdf_ORFeH_clip, gdf_ORFeH_buffered

# === Linear transport infrastructure (LTI) process === 
def process_linear_transport_infrastructure(base_dir, output_dir, extent, area_name_clean, crs_ref, gdf_ocs, gdf_ORFeH_buffered):
    """
    Processes Linear Transport Infrastructure (LTI) from BD TOPO,
    including filtering, removal of segments made passable by ORFeH wildlife crossings,
    conditional buffering, and hierarchy assignment.

    Args:
        base_dir (str): root data directory
        output_dir (str): export directory
        extent (shapely or GeoDataFrame): clipping geometry
        area_name_clean (str): cleaned EPCI name
        crs_ref (CRS): target coordinate reference system
        gdf_ocs (GeoDataFrame): OCS GE layer with CODE_US field

    Returns:
        GeoDataFrame: final hierarchical LTI layer
    """

    debug_dir = os.path.join(output_dir, "debug_LTI")
    if cfg.SAVE_VECTOR_OUTPUTS:
        os.makedirs(debug_dir, exist_ok=True)

    def load_filter_LTI(files, columns, is_road=True):
        gdfs = []
        for path in files:
            gdf = gpd.read_file(path)
            if gdf.crs != crs_ref:
                gdf = gdf.to_crs(crs_ref)
            gdf = gdf[[col for col in columns if col in gdf.columns]]
            gdf = gpd.clip(gdf, extent)
            gdf['POS_SOL'] = pd.to_numeric(gdf['POS_SOL'], errors='coerce')
            gdf = gdf[gdf['POS_SOL'] >= 0]
            if is_road:
                gdf['FICTIF'] = gdf['FICTIF'].astype(str).str.upper()
                gdf = gdf[gdf['FICTIF'] != 'OUI']
                gdf['ACCES_VL'] = gdf['ACCES_VL'].astype(str).str.upper()
                gdf['NATURE'] = gdf['NATURE'].astype(str).str.upper()
            else:
                gdf['NATURE'] = gdf['NATURE'].astype(str).str.upper()
            gdfs.append(gdf)
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=crs_ref)

# === Linear transport infrastructure (LTI) process ===
def process_linear_transport_infrastructure(base_dir, output_dir, extent, area_name_clean, crs_ref, gdf_ocs, gdf_ORFeH_buffered):
    """
    Processes Linear Transport Infrastructure (LTI) from BD TOPO,
    including filtering, removal of segments made passable by ORFeH wildlife crossings,
    conditional buffering, and hierarchy assignment.
    """

    debug_dir = os.path.join(output_dir, "debug_LTI")
    if cfg.SAVE_VECTOR_OUTPUTS:
        os.makedirs(debug_dir, exist_ok=True)

    def load_filter_LTI(files, columns, is_road=True):
        gdfs = []
        for path in files:
            gdf = gpd.read_file(path)
            if gdf.crs != crs_ref:
                gdf = gdf.to_crs(crs_ref)
            gdf = gdf[[col for col in columns if col in gdf.columns]]
            gdf = gpd.clip(gdf, extent)
            gdf['POS_SOL'] = pd.to_numeric(gdf['POS_SOL'], errors='coerce')
            gdf = gdf[gdf['POS_SOL'] >= 0]
            if is_road:
                gdf['FICTIF'] = gdf['FICTIF'].astype(str).str.upper()
                gdf = gdf[gdf['FICTIF'] != 'OUI']
                gdf['ACCES_VL'] = gdf['ACCES_VL'].astype(str).str.upper()
                gdf['NATURE'] = gdf['NATURE'].astype(str).str.upper()
            else:
                gdf['NATURE'] = gdf['NATURE'].astype(str).str.upper()
            gdfs.append(gdf)
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=crs_ref)

    def delete_undercrossing(gdf_roads, gdf_railways, gdf_crossing_buffered):
        """
        Remove road and railway segments intersecting with buffered ORFeH crossings
        of type 'passage sous ILT' (et 'passage sur ILT' inclus ici), en mappant
        explicitement les synonymes ferroviaires LGV sans toucher au mapping des routes.
        """
        # BD TOPO NATURE cibles par type
        correspondence_types = {
            "autoroute": ["TYPE AUTOROUTIER", "BRETELLE"],
            "route": ["ROUTE À 1 CHAUSSÉE", "ROUTE À 2 CHAUSSÉES", "ROND-POINT"],
            "chemin ou sentier": ["CHEMIN", "ROUTE EMPIERRÉE", "SENTIER"],
            "voie ferree": [
                "VOIE FERRÉE PRINCIPALE", "VOIE DE SERVICE", "SANS OBJET",
                "TRAMWAY", "MÉTRO", "FUNICULAIRE OU CRÉMAILLÈRE", "LGV"
            ],
            # clé spécifique LGV (séparée)
            "lgv": [
                "VOIE FERRÉE PRINCIPALE", "VOIE DE SERVICE", "SANS OBJET",
                "TRAMWAY", "MÉTRO", "FUNICULAIRE OU CRÉMAILLÈRE", "LGV"
            ],
        }

        # Synonymes / variantes pour les ROUTES uniquement (inchangé)
        merge_roads = {
            "route nationale": "route",
            "route departementale": "route",
            "autre route": "route",
        }

        # Synonymes FERROVIAIRES distincts (LGV)
        rail_synonyms = {
            "voie ferree": "voie ferree",
            "voie ferrée": "voie ferree",
            "lgv": "lgv",
            "ligne grande vitesse": "lgv",
        }

        for _, crossing in gdf_crossing_buffered.iterrows():
            crossing_type = str(crossing.get("OA_Franc_p", "")).strip().lower()
            # on accepte sous ET sur ILT (au cas où)
            if crossing_type not in ("passage sous ilt", "passage sur ilt"):
                continue

            raw_type = str(crossing.get("ILT_Type", "")).strip().lower()

            # 1) essaie d'abord la normalisation ferroviaire dédiée
            lti_type = rail_synonyms.get(raw_type)
            # 2) sinon, retombe sur les routes (comme avant)
            if lti_type is None:
                lti_type = merge_roads.get(raw_type, raw_type)

            geom = crossing.geometry
            if geom is None:
                continue

            if lti_type in correspondence_types:
                targets = correspondence_types[lti_type]

                # rails si lti_type est voie ferree OU lgv (distinct du mapping routes)
                if lti_type in ("voie ferree", "lgv"):
                    gdf_railways = gdf_railways[~(
                        (gdf_railways["NATURE"].isin(targets)) &
                        gdf_railways.intersects(geom)
                    )]
                else:
                    gdf_roads = gdf_roads[~(
                        (gdf_roads["NATURE"].isin(targets)) &
                        gdf_roads.intersects(geom)
                    )]

        return gdf_roads.reset_index(drop=True), gdf_railways.reset_index(drop=True)

    # 1. Identify departments intersecting the extent
    dept_files = [os.path.join(root, f)
                  for root, _, files in os.walk(base_dir)
                  for f in files if f.lower() == "departement.shp"]
    if not dept_files:
        raise FileNotFoundError("❌ No DEPARTEMENT layer found for LTI processing")
    gdfs = []
    for fp in dept_files:
        g = gpd.read_file(fp)
        if g.crs != crs_ref:
            g = g.to_crs(crs_ref)
        gdfs.append(g)
    gdf_departments = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True),
                                       geometry="geometry", crs=crs_ref)
    extent_geom = extent if hasattr(extent, "geom_type") else extent.unary_union
    intersect_deps = gdf_departments[gdf_departments.geometry.intersects(extent_geom)]
    if intersect_deps.empty:
        raise RuntimeError("❌ No department intersects the LTI extent")
    code_field = "INSEE_DEP" if "INSEE_DEP" in intersect_deps.columns else next(c for c in intersect_deps.columns if "DEP" in c.upper())
    codes_dep = intersect_deps[code_field].astype(str).unique().tolist()

    # 2. Collect LTI files for those departments
    roads_files, railways_files = [], []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith((".shp", ".gpkg")):
                full = os.path.join(root, f)
                if "TRONCON_DE_ROUTE" in f.upper() and any(code in full for code in codes_dep):
                    roads_files.append(full)
                elif "TRONCON_DE_VOIE_FERREE" in f.upper() and any(code in full for code in codes_dep):
                    railways_files.append(full)
    if not roads_files:
        raise RuntimeError(f"❌ LTI road data not found for departments {codes_dep}")
    if not railways_files:
        raise RuntimeError(f"❌ LTI railway data not found for departments {codes_dep}")

    # 3. Load and clip LTI data
    columns_roads = ['ID', 'NATURE', 'FICTIF', 'POS_SOL', 'ETAT', 'NB_VOIES', 'LARGEUR',
                     'ACCES_VL', 'NUMERO', 'NUM_EUROP', 'CL_ADMIN', 'TOPONYME', 'geometry']
    columns_railways = ['ID', 'NATURE', 'POS_SOL', 'ETAT', 'ELECTRIFIE', 'LARGEUR',
                        'NB_VOIES', 'USAGE', 'TOPONYME', 'geometry']
    gdf_roads = load_filter_LTI(roads_files, columns_roads, is_road=True)
    gdf_railways = load_filter_LTI(railways_files, columns_railways, is_road=False)

    # 4. Remove segments at wildlife crossings (if provided)
    if gdf_ORFeH_buffered is not None:
        gdf_roads, gdf_railways = delete_undercrossing(gdf_roads, gdf_railways, gdf_ORFeH_buffered)

    # 5. Conditional buffering of road and railway geometries
    gdf_roads['LARGEUR'] = pd.to_numeric(gdf_roads['LARGEUR'], errors='coerce')
    gdf_roads["buffer_m"] = gdf_roads["LARGEUR"].apply(lambda l: 2.5 if pd.isna(l) or l < 5 else (l / 2) + 1)
    gdf_roads["geometry"] = gdf_roads.geometry.buffer(gdf_roads["buffer_m"])

    gdf_railways['NB_VOIES'] = pd.to_numeric(gdf_railways['NB_VOIES'], errors='coerce')
    def buffered_railways(v):
        return 3.5 if pd.isna(v) or v <= 1 else min(10.5 + 2.5 * (v - 4), 20)
    gdf_railways["buffer_m"] = gdf_railways["NB_VOIES"].apply(buffered_railways)
    gdf_railways["geometry"] = gdf_railways.geometry.buffer(gdf_railways["buffer_m"])

    # 6. Hierarchical classification of LTI segments (permeability and drawing order)
    def assign_permeability(df, query, perm, draw):
        sel = df.query(query).copy()
        sel["perm"], sel["draw_order"] = perm, draw
        return sel

    non_driveable = assign_permeability(
        gdf_roads, "NATURE in ['SENTIER', 'CHEMIN'] and ACCES_VL == 'PHYSIQUEMENT IMPOSSIBLE'", 6, 6)
    zones_baties = gdf_ocs[gdf_ocs["CODE_US"].isin(["US2", "US235", "US3", "US5"])]
    if not zones_baties.empty and not non_driveable.empty:
        non_driveable = gpd.overlay(non_driveable, zones_baties.to_crs(crs_ref), how="difference")

    carrossable = assign_permeability(
        gdf_roads, "NATURE == 'ROUTE EMPIERRÉE' or (NATURE == 'CHEMIN' and ACCES_VL != 'PHYSIQUEMENT IMPOSSIBLE')", 5, 5)

    vf_autres = assign_permeability(
        gdf_railways, "NATURE in ['VOIE FERRÉE PRINCIPALE', 'VOIE DE SERVICE', 'SANS OBJET', 'TRAMWAY', 'MÉTRO', 'FUNICULAIRE OU CRÉMAILLÈRE']", 4, 4)

    routes_cl = assign_permeability(
        gdf_roads, "NATURE in ['ROUTE À 1 CHAUSSÉE', 'ROUTE À 2 CHAUSSÉES', 'ROND-POINT']", 3, 3)
    autoroutes = assign_permeability(
        gdf_roads, "NATURE in ['TYPE AUTOROUTIER', 'BRETELLE']", 2, 2)
    lgv = assign_permeability(gdf_railways, "NATURE == 'LGV'", 1, 1)

    gdf_LTI = pd.concat([non_driveable, carrossable, vf_autres, routes_cl, autoroutes, lgv], ignore_index=True)
    gdf_LTI = gdf_LTI.sort_values("draw_order").reset_index(drop=True)

    # Debug and export
    if cfg.SAVE_VECTOR_OUTPUTS:
        gdf_LTI.to_file(os.path.join(debug_dir, "LTI_Hierarchised_Debug.shp"))
    final_path = os.path.join(output_dir, f"ILT_{area_name_clean}.shp")
    if cfg.SAVE_VECTOR_OUTPUTS:
        gdf_LTI.to_file(final_path)

    return gdf_LTI

# === Fences and Solar Installations (OSM & custom data) ===
def process_fences_and_solar(base_dir, output_dir, extent, area_name_clean, crs_ref):
    """
    Processes shapefiles related to fences and solar installations (e.g., photovoltaic farms):
    - CPV_OSM_Clean.shp (solar farms from OSM): keeps FID, name, geometry
    - France_Fences_Rural.shp (rural fences): keeps all fields, applies 2.5 m buffer
    Clips both layers to the extent and exports the merged result.
    """

    gdfs = []
    path_cpv = None
    path_fences = None

    # --- Search for source files ---
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f == "CPV_OSM_Clean.shp":
                path_cpv = os.path.join(root, f)
            elif f == "France_Fences_Rural.gpkg":
                path_fences = os.path.join(root, f)

    # --- Process solar (CPV) data ---
    try:
        if path_cpv is None:
            raise FileNotFoundError("❌ File CPV_OSM_Clean.shp not found.")
        gdf_csp = gpd.read_file(path_cpv)
        if gdf_csp.crs != crs_ref:
            gdf_csp = gdf_csp.to_crs(crs_ref)
        gdf_csp = gpd.clip(gdf_csp, extent)
        gdf_csp = gdf_csp[[col for col in ["FID", "name", "geometry"] if col in gdf_csp.columns]].copy()
        gdf_csp["TYPE"] = "SOLAR"
        gdfs.append(gdf_csp)
    except Exception as e:
        print(f"⚠️ Error CPV_OSM_Clean.shp: {e}")

    # --- Process fences data ---
    try:
        if path_fences is None:
            raise FileNotFoundError("❌ File France_Fences_Rural.shp not found.")
        gdf_fences = gpd.read_file(path_fences)
        if gdf_fences.crs != crs_ref:
            gdf_fences = gdf_fences.to_crs(crs_ref)
        gdf_fences = gpd.clip(gdf_fences, extent)
        gdf_fences["geometry"] = gdf_fences.geometry.buffer(5)
        gdf_fences["TYPE"] = "FENCE"
        gdfs.append(gdf_fences)
    except Exception as e:
        print(f"⚠️ Error France_Fences_Rural.shp: {e}")

    # --- Final merge ---
    if gdfs:
        gdf_solar_fences = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), geometry="geometry", crs=crs_ref)
    else:
        gdf_solar_fences = gpd.GeoDataFrame(columns=["geometry", "TYPE"], geometry="geometry", crs=crs_ref)

    # --- Export merged layer ---
    export_path = os.path.join(output_dir, f"Solar_Fences_{area_name_clean}.shp")
    if cfg.SAVE_VECTOR_OUTPUTS:
        gdf_solar_fences.to_file(export_path)

    return gdf_solar_fences

# === Dense Built-up Areas (optional analysis) ===
def process_dense_built_areas(base_dir, output_dir, extent, area_name_clean, crs_ref, density_threshold=5):
    """
    Identifies densely built-up areas from BD TOPO 'ZONE_CONSTRUITE' and building footprints.
    Keeps zones with ≥ density_threshold buildings/km² and ≥ (density_threshold - 1) buildings.
    """
    from pathlib import Path

    # 1. Identify intersecting departments
    dept_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower() == "departement.shp":
                dept_files.append(os.path.join(root, f))
    if not dept_files:
        raise FileNotFoundError("❌ No DEPARTEMENT.shp found.")

    gdfs_dep = []
    for fp in dept_files:
        g = gpd.read_file(fp)
        if g.crs is None:
            g.set_crs(crs_ref, inplace=True)
        elif g.crs != crs_ref:
            g = g.to_crs(crs_ref)
        gdfs_dep.append(g)

    gdf_departments = gpd.GeoDataFrame(pd.concat(gdfs_dep, ignore_index=True), geometry="geometry", crs=crs_ref)
    extent_geom = extent if hasattr(extent, "geom_type") else extent.unary_union
    intersecting_deps = gdf_departments[gdf_departments.geometry.intersects(extent_geom)]
    if intersecting_deps.empty:
        raise RuntimeError("❌ No departments intersect with the buffer geometry.")
    code_field = "INSEE_DEP" if "INSEE_DEP" in intersecting_deps.columns else next(c for c in intersecting_deps.columns if "DEP" in c.upper())
    codes_dep = intersecting_deps[code_field].astype(str).unique().tolist()

    # 2. Search ZONE_CONSTRUITE files matching department codes
    zone_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith((".shp", ".gpkg")) and "ZONE_CONSTRUITE" in f.upper():
                full_path = os.path.join(root, f)
                parts = Path(full_path).parts
                if any(code in part for part in parts for code in codes_dep):
                    zone_files.append(full_path)

    if not zone_files:
        raise FileNotFoundError("❌ No ZONE_CONSTRUITE file found for the intersecting departments.")

    # 3. Load and merge constructed zones
    dense_zones = [load_gdf(path, clip_geom=extent) for path in zone_files]
    df_concat = pd.concat(dense_zones, ignore_index=True)
    geom_col = next((col for col in df_concat.columns if df_concat[col].dtype.name == "geometry"), None)
    if geom_col is None:
        raise ValueError("❌ Aucune colonne de géométrie trouvée dans les zones construites.")
    gdf_dense = gpd.GeoDataFrame(df_concat, geometry=geom_col, crs=crs_ref)
    gdf_dense = gdf_dense.dissolve().explode(index_parts=False).reset_index(drop=True)

    # 4. Load and merge building footprints
    building_paths = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower() == "batiment.shp":
                full_path = os.path.join(root, file)
                parts = Path(full_path).parts
                if any(code in part for part in parts for code in codes_dep):
                    building_paths.append(full_path)

    if not building_paths:
        raise FileNotFoundError("❌ No 'BATIMENT.shp' file found for the departments.")

    list_bati = []
    for path in building_paths:
        gdf_b = gpd.read_file(path)
        if "ETAT" in gdf_b.columns:
            gdf_b = gdf_b[gdf_b["ETAT"].str.strip().str.lower() != "en ruine"]
        if gdf_b.crs is None:
            gdf_b.set_crs(crs_ref, inplace=True)
        elif gdf_b.crs != crs_ref:
            gdf_b = gdf_b.to_crs(crs_ref)
        gdf_b = gdf_b[gdf_b.is_valid & ~gdf_b.is_empty]
        gdf_b = gdf_b[gdf_b.geometry.type.isin(["Polygon", "MultiPolygon"])]
        gdf_b = gdf_b[gdf_b.geometry.intersects(extent)].copy()
        gdf_b["geometry"] = gdf_b.geometry.intersection(extent)
        gdf_b = gdf_b[~gdf_b.is_empty & gdf_b.geometry.notnull()]
        if len(gdf_b) > 0:
            list_bati.append(gdf_b)

    if not list_bati:
        raise ValueError("❌ No valid building footprint found after filtering.")

    gdf_buildings = gpd.GeoDataFrame(pd.concat(list_bati, ignore_index=True), crs=crs_ref)

    # 5. Compute building density
    valid_zones = []
    for zone_geom in gdf_dense.geometry:
        gdf_bat_clip = gdf_buildings.clip(gpd.GeoDataFrame(geometry=[zone_geom], crs=crs_ref))
        nb_buildings = len(gdf_bat_clip)
        area_km2 = zone_geom.area / 1e6
        density = nb_buildings / area_km2 if area_km2 > 0 else 0
        if nb_buildings >= (density_threshold - 1) and density >= density_threshold:
            valid_zones.append(zone_geom)

    # 6. Export
    gdf_dense_areas = gpd.GeoDataFrame(geometry=valid_zones, crs=crs_ref)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"dense_built_zones_{area_name_clean}.shp")
    if cfg.SAVE_VECTOR_OUTPUTS:
        gdf_dense_areas.to_file(output_path)

    return gdf_dense_areas

# === Extract Habitat patches for modelisation === 
def extract_favorable_habitat_from_bdtopo(gdf_vege_clip, gdf_rpg_clip, output_path, min_area_ha=300):
    """
    Extrait les habitats favorables au cerf élaphe depuis la végétation BD TOPO,
    en nettoyant les artefacts du RPG, en retirant les zones agricoles, et en regroupant les blocs forestiers.

    Args:
        gdf_vege_clip (GeoDataFrame): BD TOPO végétation (zone_de_vegetation) découpée.
        gdf_rpg_clip (GeoDataFrame): RPG découpé à l’emprise.
        output_path (str): chemin de sortie (shapefile).
        min_area_ha (float): surface minimale d’un patch conservé (en hectares).
    """
 
    # 1. Filtrer les forêts dans la couche végétation
    filtres_vege = gdf_vege_clip['NATURE'].isin([
        'FORET_FERMEE_DE_CONIFERES',
        'FORET_FERMEE_DE_FEUILLUS',
        'FORET_FERMEE_MIXTE',
        'FORET_OUVERTE'
    ])
    gdf_forest = gdf_vege_clip[filtres_vege].copy()

    # 2. Nettoyage topologique de la couche RPG
    if not gdf_rpg_clip.empty:
        # Buffer positif
        gdf_rpg_clean = gpd.GeoDataFrame(geometry=gdf_rpg_clip.buffer(5), crs=gdf_rpg_clip.crs)
        # Dissolve
        gdf_rpg_clean = gdf_rpg_clean.dissolve()
        # Buffer négatif
        gdf_rpg_clean = gpd.GeoDataFrame(geometry=gdf_rpg_clean.buffer(-5), crs=gdf_rpg_clip.crs)

        # 3. Différence propre avec la forêt
        gdf_forest['geometry'] = gdf_forest.geometry.difference(gdf_rpg_clean.unary_union)

    # 4. Dissolve (regroupement de blocs contigus)
    gdf_forest = gdf_forest.dissolve()

    # 5. Explode (revenir à des polygones simples)
    gdf_forest = gdf_forest.explode(ignore_index=True)

    # 6. Calcul de surface + filtre
    gdf_forest['surface_ha'] = gdf_forest.geometry.area / 10_000
    gdf_forest = gdf_forest[gdf_forest['surface_ha'] >= min_area_ha]

    # 7. Export
    if gdf_forest.empty:
        print("⚠️ Aucun patch d’habitat conservé après traitement.")
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdf_forest.to_file(output_path)


