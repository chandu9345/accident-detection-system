# src/services/alert_service.py
import requests
from datetime import datetime

def send_accident_sms(to_number: str, confidence: float,
                      severity: str = None, api_key: str = None) -> dict:
    try:
        severity_line = f"Severity: {severity} | " if severity else ""

        message = (
            f"ACCIDENT DETECTED | "
            f"Confidence: {confidence * 100:.1f}% | "
            f"{severity_line}"
            f"Time: {datetime.now().strftime('%d %b %Y %I:%M %p')} | "
            f"Action: Check location immediately."
        )

        url = "https://www.fast2sms.com/dev/bulkV2"

        params = {
            "authorization" : api_key,
            "route"         : "q",
            "message"       : message,
            "flash"         : 0,
            "numbers"       : to_number
        }

        response = requests.get(url, params=params, timeout=10)
        data     = response.json()

        if data.get("return") is True:
            return {"success": True}
        else:
            return {"success": False, "error": str(data.get("message", "Unknown error"))}

    except Exception as e:
        return {"success": False, "error": str(e)}