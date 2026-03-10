"""
Market Price Predictor
Loads trained Prophet models and serves 6-month price forecasts
"""

import json, os, joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
_cache = None
_models = None

def _load():
    global _cache, _models
    p = os.path.join(MODEL_DIR, "market_forecasts_cache.json")
    if os.path.exists(p):
        with open(p) as f:
            _cache = json.load(f)
    else:
        _cache = {}
    mp = os.path.join(MODEL_DIR, "market_prophet_models.pkl")
    if os.path.exists(mp):
        _models = joblib.load(mp)
    else:
        _models = {}

def get_forecast(crop: str) -> dict:
    global _cache, _models
    if _cache is None:
        _load()
    crop = crop.lower()
    if crop not in _cache:
        return {"error": f"No forecast for '{crop}'", "available": list(_cache.keys())}

    forecast = _cache[crop]
    prices = [f['price'] for f in forecast]
    best_i  = prices.index(max(prices))
    worst_i = prices.index(min(prices))
    trend   = ((prices[-1] - prices[0]) / prices[0]) * 100

    if trend > 5:
        advice = "📈 Prices rising — consider holding stock if possible."
    elif trend < -5:
        advice = "📉 Prices falling — consider selling sooner."
    else:
        advice = "➡️ Prices stable over next 6 months."

    return {
        "crop": crop,
        "forecast": forecast,
        "best_month":  {"month": forecast[best_i]['month'],  "price": forecast[best_i]['price']},
        "worst_month": {"month": forecast[worst_i]['month'], "price": forecast[worst_i]['price']},
        "trend_pct":   round(trend, 1),
        "trend_advice": advice,
        "available_crops": list(_cache.keys()),
    }

def get_all_crops() -> list:
    if _cache is None: _load()
    return list(_cache.keys())
