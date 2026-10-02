"""
AgroSphere AI - Advisory Generator Module
Transforms quantitative surge and stockout metrics into contextual, multi-lingual dealer advisories.
"""

from typing import Dict, Any
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import PRODUCT_CATALOG, COMPANY_NAME, MANUFACTURING_PLANT
from src.genai_advisory.groq_client import GroqLLMClient
from src.genai_advisory.prompt_templates import (
    SYSTEM_ADVISORY_PROMPT, 
    FEW_SHOT_ENGLISH_TEMPLATE, 
    FEW_SHOT_GUJARATI_TEMPLATE
)

class AdvisoryGenerator:
    def __init__(self, groq_client: GroqLLMClient = None):
        self.client = groq_client or GroqLLMClient()

    def generate_advisory_text(
        self, 
        dealer_risk_row: Dict[str, Any], 
        language: str = "English"
    ) -> str:
        """
        Generates a tailored restocking advisory using Groq LLM (with fallback).
        """
        sku_id = dealer_risk_row.get("sku_id", "UB-BIO-01")
        prod_info = PRODUCT_CATALOG.get(sku_id, {})
        
        dealer_name = dealer_risk_row.get("dealer_name", "Dealer")
        contact_person = dealer_risk_row.get("contact_person", "Partner")
        district = dealer_risk_row.get("district", "Regional Hub")
        product_name = prod_info.get("name", "AgroSphere Bio-Formulation")
        active_agent = prod_info.get("active_agent", "Bio-Organic Input")
        unit = prod_info.get("unit", "Units")
        
        surge_pct = dealer_risk_row.get("surge_pct", 30.0)
        current_stock = dealer_risk_row.get("current_stock", 20)
        recommended_qty = dealer_risk_row.get("recommended_reorder_qty", 150)
        total_rain = dealer_risk_row.get("total_rain", 45.0)
        avg_humidity = dealer_risk_row.get("avg_humidity", 80.0)
        target_threats = ", ".join(prod_info.get("target_threats", ["Crop Stress"]))
        target_crops = ", ".join(prod_info.get("target_crops", ["Commercial Crops"]))
        dealer_id = dealer_risk_row.get("dealer_id", "DLR-REG-1001")
        
        # Build prompt
        user_prompt = f"""Generate a high-priority restocking advisory for an authorized agricultural dealer.

Dealer Details:
- Dealer Name: {dealer_name}
- Contact Person: {contact_person}
- Location: {district} District Hub
- Dealer ID: {dealer_id}

Product & Agro-Climatic Intelligence:
- Product: {product_name} ({active_agent})
- Target Crops & Diseases: {target_crops} | Targets: {target_threats}
- Agro-Climatic Conditions: Projected 7-day rainfall: {total_rain}mm, Average Humidity: {avg_humidity}%
- Regional Demand Surge: +{surge_pct}% over the next 14 days

Inventory Situation:
- Dealer Current Stock: {current_stock} {unit}
- Recommended Priority Reorder: {recommended_qty} {unit}
- Dispatch Facility: {MANUFACTURING_PLANT}

Requested Output Language: {language}
Include 1-click booking link: https://agrosphere.ai/dispatch/{dealer_id}
"""
        
        # If Groq client is live, generate via LLM
        if self.client.is_available():
            system_prompt = SYSTEM_ADVISORY_PROMPT + "\n\n" + (
                FEW_SHOT_GUJARATI_TEMPLATE if language.lower() == "gujarati" else FEW_SHOT_ENGLISH_TEMPLATE
            )
            return self.client.generate_completion(system_prompt, user_prompt, temperature=0.3)
            
        # High quality offline fallback template
        return self._generate_templated_advisory(
            dealer_name, contact_person, district, product_name, active_agent,
            unit, surge_pct, current_stock, recommended_qty, total_rain, 
            avg_humidity, target_threats, target_crops, dealer_id, language
        )

    def _generate_templated_advisory(
        self, dealer_name, contact_person, district, product_name, active_agent,
        unit, surge_pct, current_stock, recommended_qty, total_rain,
        avg_humidity, target_threats, target_crops, dealer_id, language
    ) -> str:
        """Deterministic, professional template generator for offline demo mode."""
        if language.lower() == "gujarati":
            return (
                f"🚨 *[એગ્રોસ્ફીયર AI - પ્રાથમિકતા રીસ્ટોક સૂચના]*\n\n"
                f"પ્રિય {contact_person} ({dealer_name}),\n"
                f"નમસ્તે.\n\n"
                f"હવામાન અહેવાલ મુજબ {district} પંથકમાં આગામી ૭ દિવસમાં {total_rain}mm વરસાદ અને {avg_humidity}% ભેજવાળું વાતાવરણ રહેવાની શક્યતા છે. આ પરિસ્થિતિ {target_crops} પાકમાં {target_threats} નું જોખમ વધારે છે.\n\n"
                f"અમારા એગ્રોસ્ફીયર AI ડિમાન્ડ મોડેલ મુજબ આપના વિસ્તારમાં **{product_name} ({active_agent})** ની માંગમાં **+{surge_pct}%** નો વધારો થવાનો અંદાજ છે.\n\n"
                f"📦 *આપના સ્ટોકનું વિશ્લેષણ:*\n"
                f"• હાલનો ઉપલબ્ધ સ્ટોક: {current_stock} {unit}\n"
                f"• ભલામણ કરેલ તાત્કાલિક રી-ઓર્ડર: *{recommended_qty} {unit}*\n\n"
                f"સમયસર સપ્લાય સુનિશ્ચિત કરવા માટે સેન્ટ્રલ પ્લાન્ટમાંથી અગાઉથી ડિસ્પેચ બુક કરો:\n"
                f"👉 [ઓર્ડર કન્ફર્મ કરો: https://agrosphere.ai/dispatch/{dealer_id}]\n\n"
                f"સાદર,\nસપ્લાય ચેઈન ટીમ, {COMPANY_NAME}"
            )
        elif language.lower() == "hindi":
            return (
                f"🚨 *[एग्रोस्फीयर AI - प्राथमिकता रीस्टॉक एडवाइजरी]*\n\n"
                f"प्रिय {contact_person} ({dealer_name}),\n"
                f"नमस्कार।\n\n"
                f"मौसम विभाग के अनुसार {district} क्षेत्र में अगले 7 दिनों में {total_rain}mm बारिश और {avg_humidity}% आर्द्रता की संभावना है। जिससे {target_crops} में {target_threats} का खतरा बढ़ सकता है।\n\n"
                f"एग्रोस्फीयर AI के अनुसार आपके क्षेत्र में **{product_name}** की मांग में **+{surge_pct}%** की भारी वृद्धि अनुमानित है।\n\n"
                f"📦 *इन्वेंट्री स्थिति:*\n"
                f"• वर्तमान स्टॉक: {current_stock} {unit}\n"
                f"• अनुशंसित री-ऑर्डर मात्रा: *{recommended_qty} {unit}*\n\n"
                f"सेंट्रल लॉजिस्टिक्स हब से प्राथमिकता प्रेषण सुरक्षित करने के लिए यहाँ क्लिक करें:\n"
                f"👉 [ऑर्डर बुक करें: https://agrosphere.ai/dispatch/{dealer_id}]\n\n"
                f"सधन्यवाद,\nसप्लाई चेन ऑटोमेशन टीम, {COMPANY_NAME}"
            )
        else: # English
            return (
                f"🚨 *[AgroSphere AI Priority Restocking Advisory]*\n\n"
                f"Dear {contact_person} ({dealer_name}),\n"
                f"Greetings from {COMPANY_NAME}.\n\n"
                f"Meteorological forecasts indicate sustained weather shifts in {district} ({total_rain}mm rainfall, {avg_humidity}% RH). This directly escalates the threat of {target_threats} across local {target_crops} cultivations.\n\n"
                f"AgroSphere AI forecasting projects a **+{surge_pct}% demand surge** for **{product_name} ({active_agent})** over the upcoming 14-day window.\n\n"
                f"📦 *Dealer Stock Status:*\n"
                f"• Current Recorded Stock: {current_stock} {unit}\n"
                f"• Recommended Early-Bird Restock: *{recommended_qty} {unit}*\n\n"
                f"Secure your pre-dispatch allocation from our Central Logistics Hub to avoid regional stockouts:\n"
                f"👉 [Confirm Fast-Track Dispatch: https://agrosphere.ai/dispatch/{dealer_id}]\n\n"
                f"Best regards,\nAI Supply Chain Automation Division, {COMPANY_NAME}"
            )
