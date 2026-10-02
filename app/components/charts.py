"""
AgroSphere AI - Dashboard Visualization Charts (Plotly)
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_demand_trend_chart(forecast_df: pd.DataFrame, selected_sku: str, selected_district: str = "All"):
    """
    Renders 14-day forward projected demand vs weather indicators.
    """
    df = forecast_df[forecast_df["sku_id"] == selected_sku].copy()
    if selected_district != "All":
        df = df[df["district"] == selected_district]
        
    daily_agg = df.groupby("date", as_index=False).agg(
        predicted_demand=("predicted_demand_qty", "sum"),
        baseline_demand=("baseline_historical_qty", "sum"),
        avg_rainfall=("precipitation_mm", "mean"),
        avg_humidity=("relative_humidity_pct", "mean")
    )
    
    fig = go.Figure()
    
    # Predicted demand bar/line
    fig.add_trace(go.Scatter(
        x=daily_agg["date"],
        y=daily_agg["predicted_demand"],
        mode="lines+markers",
        name="Projected Demand Surge",
        line=dict(color="#10B981", width=3),
        marker=dict(size=8)
    ))
    
    # Baseline comparison line
    fig.add_trace(go.Scatter(
        x=daily_agg["date"],
        y=daily_agg["baseline_demand"],
        mode="lines",
        name="Historical Baseline Demand",
        line=dict(color="#64748B", width=2, dash="dash")
    ))
    
    # Secondary axis for rainfall
    fig.add_trace(go.Bar(
        x=daily_agg["date"],
        y=daily_agg["avg_rainfall"],
        name="Rainfall Forecast (mm)",
        marker_color="rgba(59, 130, 246, 0.3)",
        yaxis="y2"
    ))
    
    fig.update_layout(
        title=f"📈 14-Day Demand Surge Forecast & Weather Signals ({selected_sku})",
        xaxis=dict(title="Forecast Date"),
        yaxis=dict(title="Demand Units"),
        yaxis2=dict(
            title="Rainfall (mm)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
        height=380
    )
    
    return fig

def create_surge_comparison_bar(regional_surges_df: pd.DataFrame):
    """
    Compares demand surge percentage across regional product categories.
    """
    df = regional_surges_df.copy()
    df["color"] = df["surge_status"].map({
        "CRITICAL_SURGE": "#EF4444",
        "HIGH_SURGE": "#F97316",
        "MODERATE_SURGE": "#FBBF24",
        "NORMAL": "#10B981",
        "DEMAND_DROP": "#6B7280"
    })
    
    fig = px.bar(
        df,
        x="district",
        y="surge_pct",
        color="product_name",
        barmode="group",
        title="📊 Projected Demand Surge % by District & Formulation",
        labels={"surge_pct": "Surge % vs Baseline", "district": "District Hub"},
        template="plotly_white",
        height=380
    )
    
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_risk_distribution_pie(actionable_dealers_df: pd.DataFrame):
    """
    Renders distribution of dealer stockout risk categories.
    """
    counts = actionable_dealers_df["risk_level"].value_counts().reset_index()
    counts.columns = ["risk_level", "count"]
    
    fig = px.pie(
        counts,
        names="risk_level",
        values="count",
        title="⚠️ Dealer Stockout Vulnerability Breakdown",
        color="risk_level",
        color_discrete_map={
            "SEVERE_STOCKOUT_RISK": "#EF4444",
            "MODERATE_STOCKOUT_RISK": "#F97316",
            "LOW_STOCK_WARNING": "#FBBF24",
            "ADEQUATELY_STOCKED": "#10B981"
        },
        hole=0.45,
        template="plotly_white",
        height=320
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig
