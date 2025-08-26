from pathlib import Path
import pandas as pd
import numpy as np

def load_instance(loc_path, dem_path, demand_site_type=None, facility_site_type=None):
    """
    Reads files and returns coords & index sets for demand nodes (I) and sites (J).
    location file columns: x, y, type
    demand file: one number per row (same order as location rows)
    """
    loc_path, dem_path = Path(loc_path), Path(dem_path)
    loc = pd.read_csv(loc_path, sep="\t", header=None, names=["x","y","type"])
    dem = pd.read_csv(dem_path, sep="\t", header=None).iloc[:,0].to_numpy(float)

    if demand_site_type is None:
        I_idx = list(range(len(loc)))
    else:
        I_idx = loc.index[loc["type"] == demand_site_type].tolist()

    if facility_site_type is None:
        J_idx = list(range(len(loc)))
    else:
        J_idx = loc.index[loc["type"] == facility_site_type].tolist()

    coords = loc[["x","y"]].to_numpy()

    return dict(
        location_df=loc,
        coords=coords,
        demand_full=dem,
        I_idx=I_idx,
        J_idx=J_idx,
        coords_I=coords[I_idx,:],
        coords_J=coords[J_idx,:],
        demand_I=dem[I_idx],
    )

