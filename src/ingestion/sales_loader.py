"""
AgroSphere AI - Sales Data Ingestion & ETL Module
Pulls, cleans, aggregates, and prepares historical Unicrop sales records for demand modeling.
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import DATA_DIR, PRODUCT_CATALOG

class SalesDataLoader:
    def __init__(self, sales_csv_path: Path = None, dealers_csv_path: Path = None):
        self.sales_path = sales_csv_path or (DATA_DIR / "raw_sales_sample.csv")
        self.dealers_path = dealers_csv_path or (DATA_DIR / "regional_dealer_directory.csv")

    def load_raw_sales(self) -> pd.DataFrame:
        """Loads and parses raw sales transactions."""
        if not self.sales_path.exists():
            raise FileNotFoundError(f"Sales dataset not found at {self.sales_path}. Run generator first.")
        
        df = pd.read_csv(self.sales_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def load_dealers(self) -> pd.DataFrame:
        """Loads the regional dealer directory."""
        if not self.dealers_path.exists():
            raise FileNotFoundError(f"Dealer dataset not found at {self.dealers_path}. Run generator first.")
        
        return pd.read_csv(self.dealers_path)

    def get_aggregated_daily_demand(self) -> pd.DataFrame:
        """
        Aggregates daily sales quantity per (date, cluster_id, district, sku_id).
        Returns a clean time-series matrix ready for feature engineering.
        """
        sales_df = self.load_raw_sales()
        
        agg_df = sales_df.groupby(
            ["date", "cluster_id", "district", "sku_id", "product_name", "category"], 
            as_index=False
        ).agg(
            total_quantity_ordered=("quantity_ordered", "sum"),
            avg_dealer_stock=("dealer_stock_on_hand", "mean"),
            active_dealer_orders=("order_id", "count")
        )
        
        # Sort chronologically
        agg_df = agg_df.sort_values(by=["cluster_id", "sku_id", "date"]).reset_index(drop=True)
        return agg_df

    def get_dealer_stock_snapshot(self) -> pd.DataFrame:
        """
        Returns the latest recorded stock on hand and reorder metrics per dealer and SKU.
        """
        sales_df = self.load_raw_sales()
        latest_date = sales_df["date"].max()
        
        # Filter to recent 7-day activity to get current snapshot
        recent_sales = sales_df[sales_df["date"] >= (latest_date - pd.Timedelta(days=7))]
        
        snapshot = recent_sales.groupby(
            ["dealer_id", "dealer_name", "district", "cluster_id", "sku_id", "product_name"],
            as_index=False
        ).agg(
            current_stock=("dealer_stock_on_hand", "last"),
            recent_order_qty=("quantity_ordered", "sum")
        )
        
        # Join with dealer contact info
        dealers_df = self.load_dealers()
        merged = pd.merge(
            snapshot,
            dealers_df[["dealer_id", "contact_person", "phone", "email", "preferred_language"]],
            on="dealer_id",
            how="left"
        )
        return merged
