import geopandas as gpd
import pandas as pd
import os
import re
import unicodedata
import config as cfg

def find_file(base_dir, keyword, extension=".shp", exact_name=False):
    for root, _, files in os.walk(base_dir):
        for file in files:
            file_lower = file.lower()
            if not file_lower.endswith(extension.lower()):
                continue
            if exact_name:
                if file_lower == f"{keyword.lower()}{extension.lower()}":
                    return os.path.join(root, file)
            else:
                if keyword.lower() in file_lower:
                    return os.path.join(root, file)
    match_type = "exactement" if exact_name else "contenant"
    raise FileNotFoundError(f"❌ Aucun fichier {match_type} '{keyword}' avec extension '{extension}' trouvé dans {base_dir}")

# Function to clean area name 
def clean_area_name(name):
    name_norm = unicodedata.normalize('NFD', name.strip())
    name_norm = ''.join(c for c in name_norm if unicodedata.category(c) != 'Mn')
    name_without_espace = re.sub(r'\s+', '_', name_norm)
    name_ascii = re.sub(r'[^a-zA-Z0-9_]', '', name_without_espace)
    return name_ascii

# Function to load a GeoDataFrame with optional clipping
def load_gdf(path, useful_columns=None, force_id=False, clip_geom=None):
    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.is_valid]
    gdf.columns = [col.upper() for col in gdf.columns]

    if "GEOMETRY" in gdf.columns:
        gdf = gdf.set_geometry("GEOMETRY")
    else:
        raise ValueError(f"No GEOMETRY column in {path}")

    if force_id and "ID" not in gdf.columns:
        gdf["ID"] = range(1, len(gdf) + 1)

    if useful_columns:
        columns_present = [col for col in useful_columns if col in gdf.columns]
        gdf = gpd.GeoDataFrame(gdf[columns_present], geometry="GEOMETRY", crs=gdf.crs)

    if clip_geom is not None:
        gdf = gdf[gdf.geometry.intersects(clip_geom)].copy()
        gdf["GEOMETRY"] = gdf.geometry.intersection(clip_geom)
        gdf = gdf[~gdf.is_empty & gdf.geometry.notnull()]

    return gdf

# === Extract Study Area ===
def extract_area_extent(base_dir, output_dir, area_name, buffer_dist=5000):
    """
    Extracts the requested EPCI, applies a buffer, and returns the buffered geometry and CRS.
    """
    # 1. Load area layer 
    path_epci = cfg.STUDY_AREA_SHAPEFILE if hasattr(cfg, "STUDY_AREA_SHAPEFILE") else find_file(base_dir, "EPCI")
    gdf_epci = gpd.read_file(path_epci)

    # 2. Find name field and filter 
    name_field = None
    for col in gdf_epci.columns:
        if col.upper() == cfg.NAME_FIELD.upper():
            name_field = col
            break
    if not name_field:
        raise ValueError(f"❌ Field '{cfg.NAME_FIELD}' not found in study area shapefile.")
    selected_area = gdf_epci[gdf_epci[name_field] == area_name]
    if selected_area.empty:
        raise ValueError(f"❌ EPCI '{area_name}' not found.")

    # 3. Clean unnecessary columns
    columns_to_drop = ["DATE_CREAT", "DATE_MAJ", "DATE_APP", "DATE_CONF", "ID_AUT_ADM"]
    selected_area = selected_area.drop(columns=columns_to_drop, errors="ignore")

    # 4. Apply Buffer 
    buffer_area = selected_area.copy()
    buffer_area["geometry"] = buffer_area.geometry.buffer(buffer_dist)

    # 5. Export buffered shapefile
    area_name_cleaned = re.sub(r'[^a-zA-Z0-9_]', '', re.sub(r'\s+', '_', area_name.strip()))
    output_buffer = os.path.join(output_dir, f"{area_name_cleaned}_Buffer_{int(buffer_dist/1000)}KM.shp")
    os.makedirs(output_dir, exist_ok=True)
    buffer_area.to_file(output_buffer)

    # 6. Create extent geometry
    geom_union = buffer_area.geometry.unary_union
    geom_extent = geom_union if hasattr(geom_union, "geom_type") else geom_union.iloc[0]

    return geom_extent, buffer_area.crs

# === Check required datasets === 
def check_required_datasets(extent, jeux, base_dir):
    """
    Vérifie que les données nécessaires sont bien présentes dans les bons départements/régions.
    """
    for jeu in jeux:
        type_donnee = jeu["type"].upper()
        niveau = jeu["niveau"].upper()

        print(f"\n🔍 Vérification des fichiers {type_donnee} ({niveau})")

        couche_admin = "DEPARTEMENT" if niveau == "DEPARTEMENT" else "REGION"
        fichiers_admin = []

        # Cherche tous les fichiers .shp pertinents
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.upper().endswith(".SHP") and couche_admin in file.upper():
                    fichiers_admin.append(os.path.join(root, file))

        if not fichiers_admin:
            raise FileNotFoundError(f"❌ Aucun fichier {couche_admin}.shp trouvé dans {base_dir}")

        # Lecture et reprojection
        gdfs = []
        for path in fichiers_admin:
            try:
                gdf = gpd.read_file(path)
                if gdf.crs != extent.crs:
                    gdf = gdf.to_crs(extent.crs)
                gdfs.append(gdf)
            except Exception as e:
                print(f"⚠️ Ignored file (not valid): {path} ({e})")

        if not gdfs:
            raise RuntimeError(f"❌ Aucun fichier {couche_admin}.shp valide trouvé.")

        gdf_admin = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=extent.crs)

        # Intersection avec l'extent
        geom_union = extent.unary_union
        gdf_admin_clip = gdf_admin[gdf_admin.geometry.intersects(geom_union)]

        champ_code = "INSEE_DEP" if niveau == "DEPARTEMENT" else "INSEE_REG"
        if champ_code not in gdf_admin_clip.columns:
            champ_code = next((c for c in gdf_admin_clip.columns if "DEP" in c.upper() or "REG" in c.upper()), None)
            if not champ_code:
                raise RuntimeError(f"❌ Aucun champ INSEE trouvé dans les fichiers {couche_admin}")

        codes = gdf_admin_clip[champ_code].astype(str).drop_duplicates().tolist()
        print(f"✅ {couche_admin}s intersectés pour {type_donnee} : {codes}")

        if len(codes) == 0:
            print(f"⚠️ Aucun {couche_admin.lower()} n’intersecte la zone d’étude.")
            continue

        # Vérification des dossiers contenant les données
        dossiers_disponibles = os.listdir(base_dir)
        def normaliser_nom(nom):
            return nom.upper().replace("-", "").replace("_", "").replace(" ", "")

        nom_jeu_normalise = normaliser_nom(type_donnee)
        dossiers_utiles = [d for d in dossiers_disponibles if nom_jeu_normalise in normaliser_nom(d)]

        # Fonction d'extraction du code selon le niveau
        def extraire_codes(nom_dossier):
            if niveau == "DEPARTEMENT":
                return re.findall(r'_D(\d{3})', nom_dossier.upper())
            elif niveau == "REGION":
                return re.findall(r'_R(\d{2})', nom_dossier.upper())
            else:
                return []

        codes_trouves = set()
        for d in dossiers_utiles:
            codes_dans_nom = extraire_codes(d)
            for code in codes:
                code_padded = code.zfill(3) if niveau == "DEPARTEMENT" else code.zfill(2)
                if code_padded in codes_dans_nom:
                    codes_trouves.add(code)

        codes_manquants = sorted(set(codes) - codes_trouves)

        if codes_manquants:
            raise FileNotFoundError(f"❌ Données manquantes pour {type_donnee} dans les {couche_admin.lower()}s : {codes_manquants}")
        else:
            print(f"✅ Toutes les données nécessaires pour {type_donnee} sont présentes.")


