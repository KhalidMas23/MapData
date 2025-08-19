# app.py — local-data-only, robust layer handling (numpy arrays, name/index, spatial & non-spatial)
import os
from typing import List, Tuple, Any

import json
import pydeck as pdk
import streamlit as st
import pandas as pd
import geopandas as gpd
from pyogrio import list_layers as og_list_layers, read_dataframe as og_read_dataframe
from shapely import make_valid
import numpy as np  # <-- important for numpy.ndarray handling

st.set_page_config(page_title="Environmental Data Viewer", layout="wide")
st.title("Environmental Data Viewer")

DATA_DIR = os.environ.get("DATA_DIR", "data")
REQUIRED_CATEGORIES = ["contamination", "demographic", "environmental", "stress"]

# ---------------------------- helpers ----------------------------
def safe_list_categories(base_dir: str, expected: List[str]) -> List[str]:
    return [d for d in expected if os.path.isdir(os.path.join(base_dir, d))]

def list_gpkg_files(cat_path: str) -> List[str]:
    return sorted([f for f in os.listdir(cat_path) if f.lower().endswith(".gpkg")])

def raw_list_layers(path: str):
    return og_list_layers(path)  # may return tuples/lists/ndarrays

def normalize_layers(raw_layers) -> Tuple[pd.DataFrame, List[str], dict]:
    """
    Normalize pyogrio.list_layers() output to a consistent DataFrame and
    return: (df, layer_names, name_to_index)
    Each row: idx, name, geom, features, srs, is_spatial
    """
    rows = []
    name_to_idx = {}
    seq_types = (list, tuple, np.ndarray)

    for i, li in enumerate(raw_layers):
        if isinstance(li, seq_types):
            # Expected shapes:
            #  - (name,)
            #  - (name, geom)
            #  - (name, geom, features)
            #  - (name, geom, features, srs)
            name = li[0] if len(li) > 0 else f"layer_{i}"
            geom = li[1] if len(li) > 1 else None
            features = li[2] if len(li) > 2 else None
            srs = li[3] if len(li) > 3 else None
        else:
            # Unexpected scalar/object: stringify safely
            name, geom, features, srs = str(li), None, None, None

        is_spatial = geom not in (None, "None", "", "Unknown")
        row = {
            "idx": i,
            "name": str(name),
            "geom": str(geom) if geom is not None else "None",
            "features": features,
            "srs": str(srs) if srs else "None",
            "is_spatial": bool(is_spatial),
        }
        rows.append(row)
        name_to_idx[row["name"]] = i

    df = pd.DataFrame(rows, columns=["idx", "name", "geom", "features", "srs", "is_spatial"])
    return df, df["name"].tolist(), name_to_idx

def read_layer(path: str, layer_name: str, layer_index: int, is_spatial: bool) -> gpd.GeoDataFrame:
    """
    Try reading by name; if that fails, read by index.
    For non-spatial layers, pass read_geometry=False.
    """
    # 1) try by name
    try:
        if is_spatial:
            return og_read_dataframe(path, layer=layer_name)
        else:
            return og_read_dataframe(path, layer=layer_name, read_geometry=False)
    except Exception as e_name:
        # 2) fallback to index
        try:
            if is_spatial:
                return og_read_dataframe(path, layer=layer_index)
            else:
                return og_read_dataframe(path, layer=layer_index, read_geometry=False)
        except Exception as e_idx:
            # 3) final fallback via geopandas using engine='pyogrio'
            try:
                if is_spatial:
                    return gpd.read_file(path, layer=layer_name, engine="pyogrio")
                else:
                    df = og_read_dataframe(path, layer=layer_index, read_geometry=False)
                    return gpd.GeoDataFrame(df, geometry=None, crs=None)
            except Exception as e_gp:
                raise RuntimeError(
                    f"Failed to read layer '{layer_name}' (idx {layer_index}). "
                    f"name_err={e_name}; idx_err={e_idx}; gp_err={e_gp}"
                ) from e_gp

def heal_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fix invalid geometries using make_valid, then buffer(0) per-row if needed."""
    geom_col = getattr(gdf, "geometry", None)
    if gdf.empty or geom_col is None:
        return gdf
    try:
        invalid = ~gdf.geometry.is_valid
        if invalid.any():
            gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)
        still_invalid = ~gdf.geometry.is_valid
        if still_invalid.any():
            gdf.loc[still_invalid, "geometry"] = gdf.loc[still_invalid, "geometry"].buffer(0)
    except Exception as e:
        st.info(f"Geometry validation skipped: {e}")
    return gdf

def arrow_safe_preview(df: Any, n: int = 200):
    """Show a table preview that won't crash Arrow serialization."""
    if isinstance(df, gpd.GeoDataFrame):
        try:
            st.dataframe(df.head(n).drop(columns=df.geometry.name, errors="ignore"))
            return
        except Exception:
            prev = df.head(n).copy()
            if "geometry" in prev.columns:
                prev["geometry_wkt"] = prev.geometry.to_wkt()
                prev = prev.drop(columns=["geometry"])
            st.dataframe(prev)
    else:
        st.dataframe(df.head(n))

def map_points(gdf: gpd.GeoDataFrame):
    """Simple point mapping (polygons/lines not rendered here)."""
    try:
        if getattr(gdf, "geometry", None) is None:
            st.info("Non-spatial layer (no geometry). Table shown above.")
            return
        gdf_ll = gdf.to_crs(4326)
        if hasattr(gdf_ll, "geom_type") and gdf_ll.geom_type.isin(["Point"]).all():
            pts = gdf_ll.copy()
            pts["latitude"] = pts.geometry.y
            pts["longitude"] = pts.geometry.x
            st.map(pts[["latitude", "longitude"]])
        else:
            st.info("Selected layer is not points; polygon/line rendering is limited in this viewer.")
    except Exception as e:
        st.warning(f"Map preview skipped: {e}")


def map_geoms(gdf):
    """Render GeoDataFrame on an interactive map (points/lines/polygons)."""
    if getattr(gdf, "geometry", None) is None or gdf.empty:
        st.info("No geometry to map for this layer.")
        return

    # Reproject to WGS84
    gdf_ll = gdf.to_crs(4326)

    # Compute a sensible view (center on bounds)
    minx, miny, maxx, maxy = gdf_ll.total_bounds
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    # crude zoom guess by extent (smaller extent → higher zoom)
    dx = max(1e-6, maxx - minx)
    dy = max(1e-6, maxy - miny)
    extent = max(dx, dy)
    # map extent (~360 deg world) → zoom 0; ~0.01 deg neighborhood → zoom ~14
    import math
    zoom = max(1, min(14, 12 - math.log10(extent + 1e-12)))

    view = pdk.ViewState(latitude=cy, longitude=cx, zoom=zoom, bearing=0, pitch=0)

    # Choose layer by geometry type
    geom_types = set(gdf_ll.geom_type.unique())
    layers = []

    if geom_types <= {"Point"}:
        pts = gdf_ll.copy()
        pts["lon"] = pts.geometry.x
        pts["lat"] = pts.geometry.y
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=pts[["lon", "lat"]],
                get_position="[lon, lat]",
                get_radius=30,
                pickable=False,
            )
        )
    else:
        # Use GeoJSONLayer for lines/polygons (and mixed)
        geojson = json.loads(gdf_ll.to_json())
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                data=geojson,
                stroked=True,
                filled=True,
                get_line_width=2,
                line_width_min_pixels=1,
                # (Optional colors; remove if you prefer Deck defaults)
                get_fill_color=[200, 120, 80, 60],
                get_line_color=[50, 50, 50, 255],
                pickable=False,
            )
        )

    r = pdk.Deck(map_style="light", initial_view_state=view, layers=layers)
    st.pydeck_chart(r)

def load_layer(path, layer_name, layer_index, is_spatial):
    """Your existing robust reader, returning a GeoDataFrame (or DataFrame for non-spatial)."""
    return read_layer(path, layer_name, layer_index, is_spatial)

def _ensure_crs_match(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame, to_epsg=4326):
    """Project both to the same CRS (defaults to WGS84) to make predicates reliable."""
    if a.crs is None and b.crs is None:
        a = a.set_crs(epsg=to_epsg)
        b = b.set_crs(epsg=to_epsg)
    elif a.crs is None:
        a = a.set_crs(b.crs)
    elif b.crs is None:
        b = b.set_crs(a.crs)

    if a.crs != b.crs:
        # choose projected CRS for area/length if needed; otherwise align to a.crs
        b = b.to_crs(a.crs)
    return a, b

def compute_overlap(a_gdf: gpd.GeoDataFrame, b_gdf: gpd.GeoDataFrame, mode: str, distance_m: float | None = None):
    """
    mode:
      - 'polygon-polygon' → exact overlap geometries (intersection)
      - 'point-in-polygon' → points of A within polygons of B
      - 'line-polygon' → line segments of A intersecting polygons of B (clipped)
      - 'intersects' → general intersects (returns A rows with B hit attributes)
      - 'point-near-point' (requires distance_m) → points of A within distance of points of B
    """
    a_gdf, b_gdf = _ensure_crs_match(a_gdf, b_gdf)

    # Speed up with spatial index
    # geopandas uses shapely 2 STRtree under the hood for sjoin/overlay
    if mode == "polygon-polygon":
        # return the *overlap geometry*, not just attributes
        return gpd.overlay(a_gdf, b_gdf, how="intersection", keep_geom_type=True)

    if mode == "point-in-polygon":
        return gpd.sjoin(a_gdf, b_gdf, predicate="within", how="inner")

    if mode == "line-polygon":
        # clip lines to polygons to get intersecting segments
        return gpd.overlay(a_gdf, b_gdf, how="intersection", keep_geom_type=True)

    if mode == "intersects":
        return gpd.sjoin(a_gdf, b_gdf, predicate="intersects", how="inner")

    if mode == "point-near-point":
        if distance_m is None:
            raise ValueError("distance_m is required for point-near-point")
        # Buffer B by distance, then point-in-polygon
        b_buf = b_gdf.copy()
        # if CRS is geographic, distance is in degrees; for accuracy, you may want to reproject to a metric CRS
        # quick heuristic: if geographic, switch to EPSG:3857 for a local-scale metric
        if a_gdf.crs and a_gdf.crs.is_geographic:
            a_gdf = a_gdf.to_crs(3857)
            b_buf = b_buf.to_crs(3857)
        b_buf["geometry"] = b_buf.geometry.buffer(distance_m)
        out = gpd.sjoin(a_gdf, b_buf, predicate="within", how="inner")
        # return to original CRS for display
        if out.crs and out.crs.to_epsg() == 3857:
            out = out.to_crs(4326)
        return out

    raise ValueError(f"Unknown mode: {mode}")

def summarize_overlap(gdf: gpd.GeoDataFrame):
    """Return a small dict of stats (rows; area if polygon; length if line)."""
    stats = {"rows": len(gdf)}
    try:
        if not gdf.empty and hasattr(gdf, "geometry") and gdf.geometry.iloc[0] is not None:
            # If projected in meters, area/length are meaningful
            proj = gdf
            if proj.crs is None or proj.crs.is_geographic:
                # Web Mercator for approximate metrics
                proj = gdf.to_crs(3857)
            geom_type = proj.geom_type.unique().tolist()
            if any("Polygon" in gt for gt in geom_type):
                stats["area_m2"] = float(proj.area.sum())
            if any("LineString" in gt for gt in geom_type):
                stats["length_m"] = float(proj.length.sum())
    except Exception:
        pass
    return stats

def download_buttons(gdf: gpd.GeoDataFrame, label_prefix="overlap"):
    """Offer downloads as GeoJSON and (if desired) GeoPackage."""
    if gdf.empty:
        return
    # GeoJSON
    geojson_str = gdf.to_crs(4326).to_json()
    st.download_button(
        "⬇️ Download overlap (GeoJSON)",
        data=geojson_str.encode("utf-8"),
        file_name=f"{label_prefix}.geojson",
        mime="application/geo+json",
    )
    # GPKG (write to temp then serve bytes)
    try:
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, f"{label_prefix}.gpkg")
            gdf.to_file(out_path, driver="GPKG")
            with open(out_path, "rb") as f:
                st.download_button(
                    "⬇️ Download overlap (GeoPackage)",
                    data=f.read(),
                    file_name=f"{label_prefix}.gpkg",
                    mime="application/octet-stream",
                )
    except Exception as e:
        st.info(f"GPKG export skipped: {e}")
# ---------------------------- sanity checks ----------------------------
if not os.path.isdir(DATA_DIR):
    st.error(f"Missing data directory: '{DATA_DIR}'. Place your folders under `{DATA_DIR}/…` and redeploy.")
    st.stop()

categories = safe_list_categories(DATA_DIR, REQUIRED_CATEGORIES)
if not categories:
    st.error(f"No category folders found under '{DATA_DIR}'. Expected: {REQUIRED_CATEGORIES}.")
    st.stop()

# ---------------------------- sidebar selection ----------------------------
with st.sidebar:
    st.header("Browse")
    # ensure we always define these for the main view
    _browse_ok = True

    try:
        category = st.selectbox("Category", categories, key="browse_cat")
        cat_path = os.path.join(DATA_DIR, category)

        gpkg_files = list_gpkg_files(cat_path)
        if not gpkg_files:
            st.warning(f"No .gpkg files in {cat_path}")
            _browse_ok = False
        else:
            file_choice = st.selectbox("GeoPackage file", gpkg_files, key="browse_file")
            file_path = os.path.join(cat_path, file_choice)

            layers_df, layer_names, name_to_index = normalize_layers(raw_list_layers(file_path))
            layer = st.selectbox("Layer", layer_names, key="browse_layer")
    except Exception as _e:
        _browse_ok = False
        st.error(f"Browse panel failed: {_e}")

    st.divider()
    st.header("Overlap")
    st.caption("Compare two layers (can be from different files).")

    # --- A selection ---
    catA = st.selectbox("Category A", categories, key="ov_catA")
    pathA = os.path.join(DATA_DIR, catA)
    filesA = list_gpkg_files(pathA)
    fileA = st.selectbox("File A", filesA, key="ov_fileA")
    fpathA = os.path.join(pathA, fileA)
    layers_dfA, layer_namesA, name_to_indexA = normalize_layers(raw_list_layers(fpathA))
    layerA = st.selectbox("Layer A", layer_namesA, key="ov_layerA")

    # --- B selection ---
    catB = st.selectbox("Category B", categories, key="ov_catB")
    pathB = os.path.join(DATA_DIR, catB)
    filesB = list_gpkg_files(pathB)
    fileB = st.selectbox("File B", filesB, key="ov_fileB")
    fpathB = os.path.join(pathB, fileB)
    layers_dfB, layer_namesB, name_to_indexB = normalize_layers(raw_list_layers(fpathB))
    layerB = st.selectbox("Layer B", layer_namesB, key="ov_layerB")

    mode = st.selectbox(
        "Overlap mode",
        ["polygon-polygon", "point-in-polygon", "line-polygon", "intersects", "point-near-point"],
        key="ov_mode",
    )
    distance_m = st.number_input("Distance (meters)", min_value=1.0, value=250.0, step=50.0, key="ov_dist") \
        if mode == "point-near-point" else None
    run_overlap = st.button("Compute overlap", key="ov_run")


# ---------------------------- main ----------------------------
# ---------------------------- main (browse) ----------------------------
if _browse_ok:
    st.subheader(f"File: {file_choice}")
    st.caption(f"Path: `{os.path.join(category, file_choice)}` — Layer: `{layer}`")

    with st.expander("Layer metadata", expanded=False):
        st.dataframe(layers_df)

    layer_index = name_to_index.get(layer, 0)
    layer_row = layers_df.loc[layers_df["name"] == layer].iloc[0] if layer in layers_df["name"].values else None
    is_spatial = bool(layer_row["is_spatial"]) if layer_row is not None else True

    gdf = read_layer(file_path, layer, layer_index, is_spatial)
    if is_spatial and isinstance(gdf, gpd.GeoDataFrame):
        gdf = heal_geometries(gdf)

    if isinstance(gdf, gpd.GeoDataFrame):
        st.write(f"CRS: **{gdf.crs}**")
    st.write(f"Rows: **{len(gdf)}** | Columns: **{len(gdf.columns)}**")

    arrow_safe_preview(gdf, n=200)
    # use your pydeck renderer
    map_geoms(gdf)
else:
    st.info("Use the **Browse** panel to pick a file/layer.")
