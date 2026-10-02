"""
AgroSphere AI - Regional Map Component for Gujarat Clusters
"""

import plotly.express as px
import pandas as pd
from src.config import REGIONAL_CLUSTERS

def create_gujarat_cluster_map(regional_surges_df: pd.DataFrame):
    """
    Renders an interactive map of Gujarat regional clusters with surge intensity markers.
    """
    map_data = []
    
    for cluster_id, info in REGIONAL_CLUSTERS.items():
        sub_df = regional_surges_df[regional_surges_df["cluster_id"] == cluster_id]
        max_surge = sub_df["surge_pct"].max() if len(sub_df) > 0 else 0.0
        alert_count = int(sub_df["is_alert_triggered"].sum()) if len(sub_df) > 0 else 0
        
        map_data.append({
            "Cluster": cluster_id,
            "District": info["district"],
            "lat": info["lat"],
            "lon": info["lon"],
            "Dealers": info["dealer_count"],
            "Primary Crops": ", ".join(info["primary_crops"][:2]),
            "Max Surge %": max_surge,
            "Critical Alerts": alert_count,
        })
        
    df_map = pd.DataFrame(map_data)
    
    fig = px.scatter_mapbox(
        df_map,
        lat="lat",
        lon="lon",
        size="Dealers",
        color="Max Surge %",
        hover_name="District",
        hover_data=["Primary Crops", "Dealers", "Max Surge %", "Critical Alerts"],
        color_continuous_scale="Viridis",
        size_max=35,
        zoom=6.5,
        center={"lat": 22.2587, "lon": 71.1924},
        mapbox_style="carto-positron",
        title="📍 Gujarat Distribution Hubs & Surge Heatmap (Vadodara HQ Hub)"
    )
    
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        height=380
    )
    
    return fig
