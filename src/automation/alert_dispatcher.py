"""
AgroSphere AI - Alert Dispatcher & Business Process Automation
Simulates WhatsApp Business API, Email Webhooks, and SMS dispatch workflows for dealers.
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import OUTPUTS_DIR, COMPANY_NAME
from src.genai_advisory.advisory_generator import AdvisoryGenerator

class AlertDispatcher:
    def __init__(self, advisory_gen: AdvisoryGenerator = None):
        self.advisory_gen = advisory_gen or AdvisoryGenerator()
        self.dispatch_log_path = OUTPUTS_DIR / "alert_dispatch_log.json"

    def dispatch_dealer_alerts(
        self, 
        actionable_dealers_df: pd.DataFrame, 
        max_dispatches: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Dispatches multi-lingual automated restocking alerts to flagged high-risk dealers.
        """
        records = actionable_dealers_df.head(max_dispatches).to_dict(orient="records")
        dispatched_logs = []
        
        for item in records:
            lang = item.get("preferred_language", "English")
            message_body = self.advisory_gen.generate_advisory_text(item, language=lang)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dealer_id = item.get("dealer_id")
            dealer_name = item.get("dealer_name")
            phone = item.get("phone")
            email = item.get("email")
            sku_id = item.get("sku_id")
            product_name = item.get("product_name")
            reorder_qty = item.get("recommended_reorder_qty")
            risk_level = item.get("risk_level")
            
            # Formulate simulated WhatsApp / Webhook Payload
            payload = {
                "dispatch_id": f"DSP-GJ-{len(dispatched_logs) + 10001}",
                "timestamp": timestamp,
                "dealer_id": dealer_id,
                "dealer_name": dealer_name,
                "recipient_phone": phone,
                "recipient_email": email,
                "preferred_language": lang,
                "risk_level": risk_level,
                "sku_id": sku_id,
                "product_name": product_name,
                "recommended_reorder_qty": reorder_qty,
                "channels": ["WhatsApp_Business_API", "Email_PO", "SMS_Gateway"],
                "status": "DELIVERED_200_OK",
                "message_body": message_body,
                "order_tracking_url": f"https://agrosphere.ai/dispatch/{dealer_id}?sku={sku_id}&qty={reorder_qty}",
            }
            
            dispatched_logs.append(payload)
            
        # Save dispatch logs to JSON and CSV
        with open(self.dispatch_log_path, "w", encoding="utf-8") as f:
            json.dump(dispatched_logs, f, indent=2, ensure_ascii=False)
            
        df_logs = pd.DataFrame(dispatched_logs)
        df_logs.to_csv(OUTPUTS_DIR / "alert_dispatch_log.csv", index=False)
        
        return dispatched_logs

    def load_recent_dispatch_logs(self) -> List[Dict[str, Any]]:
        """Loads historical dispatch logs from file."""
        if self.dispatch_log_path.exists():
            with open(self.dispatch_log_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
