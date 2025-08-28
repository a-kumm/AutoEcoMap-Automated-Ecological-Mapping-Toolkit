import warnings
warnings.filterwarnings("ignore")

import os
import geopandas as gpd
import config as cfg
from tqdm import tqdm
from utils_ocs_final import clean_area_name, check_required_datasets
from data_processing_final import (
    process_land_cover_data,
    process_rpg_data,
    process_vegetation_data,
    process_hydrography_network,
    process_technical_infrastructure,
    process_wildlife_crossing,
    process_linear_transport_infrastructure,
    process_fences_and_solar,
    process_dense_built_areas,
    extract_favorable_habitat_from_bdtopo
)
from raster_processing_final import (
    load_table_from_csv,
    rasterize_classes_and_friction,
    process_dtm_from_tiles,
    apply_slope_weighting,
    apply_building_distance_weighting,
    replace_obstacle_friction_by_local_interp,
    replace_class3_1000_by_local_mode
)

if __name__ == '__main__':
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    epci_shp_path = cfg.STUDY_AREA_SHAPEFILE
    if not os.path.isfile(epci_shp_path):
        raise FileNotFoundError(f"Study area shapefile not found: {epci_shp_path}")
    gdf_epci = gpd.read_file(epci_shp_path)

    name_field = next(
        (col for col in gdf_epci.columns if col.upper() == cfg.NAME_FIELD.upper()),
        next((col for col in gdf_epci.columns if cfg.NAME_FIELD.upper() in col.upper()), None)
    )
    if not name_field:
        raise ValueError(f"❌ Field '{cfg.NAME_FIELD}' not found in study area shapefile.")

    all_names = gdf_epci[name_field].astype(str).unique().tolist()
    failures = {}

    for area_name in all_names:
        try:
            with tqdm(total=19, desc=f"[{area_name}] Starting", leave=True, ncols=100) as pbar:
                area_name_clean = clean_area_name(area_name)
                area_output_dir = os.path.join(cfg.OUTPUT_DIR, area_name_clean)
                os.makedirs(area_output_dir, exist_ok=True)

                selected_area = gdf_epci[gdf_epci[name_field].astype(str).str.strip().str.upper() == area_name.strip().upper()]
                if selected_area.empty:
                    raise ValueError(f"EPCI '{area_name}' not found.")

                buffer_area = selected_area.copy()
                buffer_area["geometry"] = buffer_area.geometry.buffer(cfg.BUFFER_DISTANCE)
                buffer_area.set_crs(selected_area.crs, inplace=True)

                geom_extent = buffer_area.geometry.unary_union
                crs_ref = buffer_area.crs

                buffer_geojson_path = os.path.join(area_output_dir, f"{area_name_clean}_Buffer_{int(cfg.BUFFER_DISTANCE/1000)}KM.geojson")
                buffer_area.to_file(buffer_geojson_path, driver="GeoJSON")

                extent_gdf = gpd.GeoDataFrame({'geometry': gpd.GeoSeries([geom_extent])}, crs=crs_ref)
                check_required_datasets(
                    extent=extent_gdf,
                    jeux=[
                        {"type": "BD TOPO", "niveau": "DEPARTEMENT"},
                        {"type": "OCS GE", "niveau": "DEPARTEMENT"},
                        {"type": "RPG", "niveau": "REGION"}
                    ],
                    base_dir=cfg.BASE_DIR
                )
                pbar.set_description("Buffered area exported")
                pbar.update(1)
                pbar.set_description("Datasets checked")
                pbar.update(1)

                gdf_ocs = process_land_cover_data(cfg.BASE_DIR, area_output_dir, geom_extent, area_name_clean, cfg.NUM_WORKERS)
                pbar.set_description("OCS GE processed")
                pbar.update(1)

                gdf_rpg = process_rpg_data(cfg.BASE_DIR, area_output_dir, geom_extent, area_name_clean, crs_ref)
                gdf_vegetation, gdf_hedges = process_vegetation_data(cfg.BASE_DIR, area_output_dir, geom_extent, area_name_clean, crs_ref)
                pbar.set_description("RPG & vegetation processed")
                pbar.update(1)

                gdf_hydro = process_hydrography_network(cfg.BASE_DIR, area_output_dir, geom_extent, area_name_clean, crs_ref)
                pbar.set_description("Hydro processed")
                pbar.update(1)

                gdf_tech_infra = process_technical_infrastructure(cfg.BASE_DIR, area_output_dir, geom_extent, area_name_clean, crs_ref)
                gdf_wildlife_crossings, gdf_wildlife_buffer = process_wildlife_crossing(
                    cfg.BASE_DIR, area_output_dir, geom_extent, crs_ref, area_name_clean
                )
                gdf_lti_final = process_linear_transport_infrastructure(
                    cfg.BASE_DIR, area_output_dir, geom_extent, area_name_clean, crs_ref, gdf_ocs, gdf_wildlife_buffer
                )
                pbar.set_description("Infrastructure processed")
                pbar.update(1)

                gdf_solar_fences = process_fences_and_solar(cfg.BASE_DIR, area_output_dir, geom_extent, area_name_clean, crs_ref)
                gdf_dense_area = process_dense_built_areas(cfg.BASE_DIR, area_output_dir, geom_extent, area_name_clean, crs_ref)
                pbar.set_description("Fences & built-up processed")
                pbar.update(1)

                habitat_output_path = os.path.join(area_output_dir, f"{area_name_clean}_habitat_bdtopo_cleaned.shp")
                extract_favorable_habitat_from_bdtopo(gdf_vegetation, gdf_rpg, habitat_output_path, min_area_ha=300)
                pbar.set_description("Extract Habitat patches")
                pbar.update(1)

                # === Raster classe & friction (base) ===
                df_table = load_table_from_csv(
                    os.path.join(cfg.BASE_DIR, "Table_Raster.csv"),
                    {
                        "OCS": gdf_ocs, "RPG": gdf_rpg, "VEGETATION": gdf_vegetation,
                        "HEDGES": gdf_hedges, "HYDRO": gdf_hydro, "LTI": gdf_lti_final,
                        "SOLAR_FENCES": gdf_solar_fences, "TECH_INFRA": gdf_tech_infra,
                        "BUILT_AREA": gdf_dense_area
                    }
                )
                pbar.set_description("Classification with all resistance table loaded")
                pbar.update(1)

                rasterize_classes_and_friction(
                    table_df=df_table,
                    vector_layers={
                        "OCS": gdf_ocs, "RPG": gdf_rpg, "VEGETATION": gdf_vegetation,
                        "HEDGES": gdf_hedges, "HYDRO": gdf_hydro, "LTI": gdf_lti_final,
                        "SOLAR_FENCES": gdf_solar_fences, "TECH_INFRA": gdf_tech_infra,
                        "BUILT_AREA": gdf_dense_area
                    },
                    extent=geom_extent,
                    crs_ref=crs_ref,
                    resolution=cfg.RESOLUTION,
                    output_dir=area_output_dir,
                    area_name_clean=area_name_clean + "_full"
                )
                pbar.set_description("Rasterized map with all resistance")
                pbar.update(1)

                # === DTM & pondération pente/bâti ===
                path_dtm = os.path.join(area_output_dir, f"DEM_{area_name_clean}.tif")
                process_dtm_from_tiles(geom_extent, crs_ref, cfg.BASE_DIR, path_dtm)
                pbar.set_description("DTM processed")
                pbar.update(1)

                friction_path = os.path.join(area_output_dir, f"Raster_Friction_{area_name_clean}_full.tif")
                friction_slope_path = os.path.join(area_output_dir, f"Raster_Friction_Slope_{area_name_clean}_full.tif")
                apply_slope_weighting(friction_path, path_dtm, friction_slope_path)

                final_friction_base_path = os.path.join(area_output_dir, f"Raster_Friction_Final_{area_name_clean}_full.tif")
                path_raster_classes = os.path.join(area_output_dir, f"Raster_Classe_{area_name_clean}_full.tif")
                apply_building_distance_weighting(friction_slope_path, path_raster_classes, cfg.BUILDING_CLASS_CODE, final_friction_base_path)
                pbar.set_description("Final friction with all resistance rasterized")
                pbar.update(1)

                final_friction_base_path = replace_class3_1000_by_local_mode(
                    path_friction=final_friction_base_path,
                    path_classes=path_raster_classes,
                    output_path=final_friction_base_path,  # overwrite en place
                    window_size=11,                        # 11x11 comme ta fonction scénario
                    nodata_value=0
                )

                pbar.set_description("Full raster fixed (class 3 @1000 → local mode)")
                pbar.update(1)

                # === SCÉNARIOS (base = raster FULL déjà corrigé) ===
                friction_full_path = final_friction_base_path
                class_full_path = path_raster_classes

                # 1) no_fences → neutraliser clôtures
                nf_path = friction_full_path.replace("_full.tif", "_no_fences.tif")
                replace_obstacle_friction_by_local_interp(
                    path_friction=friction_full_path,
                    path_classes=class_full_path,
                    target_class_codes=[39, 40],
                    path_output=nf_path
                )

                # 2) only_fences → neutraliser ILT (sans 3)
                of_path = friction_full_path.replace("_full.tif", "_only_fences.tif")
                replace_obstacle_friction_by_local_interp(
                    path_friction=friction_full_path,
                    path_classes=class_full_path,
                    target_class_codes=[35, 36, 37, 38],
                    path_output=of_path
                )

                # 3) no_permeability → neutraliser clôtures + ILT
                np_path = friction_full_path.replace("_full.tif", "_no_permeability.tif")
                replace_obstacle_friction_by_local_interp(
                    path_friction=friction_full_path,
                    path_classes=class_full_path,
                    target_class_codes=[35, 36, 37, 38, 39, 40],
                    path_output=np_path
                )

                pbar.set_description("Completed")
                pbar.update(1)
                pbar.refresh()

        except Exception as e:
            failures[area_name] = str(e)
            print(f"❌ Error processing '{area_name}': {e}")

    if failures:
        failed_names = ', '.join(failures.keys())
        print(f"\n⚠️ Pipeline finished with errors for {len(failures)} out of {len(all_names)} areas: {failed_names}")
    else:
        print(f"\n✅ Pipeline successfully completed for all {len(all_names)} areas.")
