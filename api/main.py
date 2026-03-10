"""
Smart Crop Advisor v2 - FastAPI Backend
All API routes under /api/ prefix. No StaticFiles mount.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np, joblib, json, os, io, httpx
from PIL import Image
import tensorflow as tf

from market_predictor import get_forecast, get_all_crops
from weather_forecast  import get_7day_forecast
from translator        import translate_text, get_supported_languages
from chatbot           import get_chat_response

app = FastAPI(title="Smart Crop Advisor API v2", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=True,
)

BASE         = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE, "../frontend"))
MODEL_DIR    = os.path.abspath(os.path.join(BASE, "../models"))

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/index.html")
async def serve_index_html():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/auth.html")
async def serve_auth():
    return FileResponse(os.path.join(FRONTEND_DIR, "auth.html"))

@app.get("/manifest.json")
async def serve_manifest():
    p = os.path.join(FRONTEND_DIR, "manifest.json")
    return FileResponse(p) if os.path.exists(p) else JSONResponse({"name": "Kisan AI"})

crop_model = crop_scaler = crop_le = None
fert_model = fert_scaler = fert_le_soil = fert_le_crop = fert_le_fert = None
disease_model = None
disease_classes = []

@app.on_event("startup")
async def load_models():
    global crop_model, crop_scaler, crop_le
    global fert_model, fert_scaler, fert_le_soil, fert_le_crop, fert_le_fert
    global disease_model, disease_classes

    crop_model  = joblib.load(f"{MODEL_DIR}/crop_model.pkl")
    crop_scaler = joblib.load(f"{MODEL_DIR}/crop_scaler.pkl")
    crop_le     = joblib.load(f"{MODEL_DIR}/crop_label_encoder.pkl")
    fert_model   = joblib.load(f"{MODEL_DIR}/fertilizer_model.pkl")
    fert_scaler  = joblib.load(f"{MODEL_DIR}/fertilizer_scaler.pkl")
    fert_le_soil = joblib.load(f"{MODEL_DIR}/fertilizer_le_soil.pkl")
    fert_le_crop = joblib.load(f"{MODEL_DIR}/fertilizer_le_crop.pkl")
    fert_le_fert = joblib.load(f"{MODEL_DIR}/fertilizer_le_fert.pkl")

    keras_path = f"{MODEL_DIR}/disease_model.keras"
    h5_path    = f"{MODEL_DIR}/disease_model.h5"
    dis_path   = keras_path if os.path.exists(keras_path) else (h5_path if os.path.exists(h5_path) else None)
    if dis_path:
        try:
            disease_model = tf.keras.models.load_model(dis_path)
            idx_json = f"{MODEL_DIR}/disease_class_indices.json"
            if os.path.exists(idx_json):
                with open(idx_json) as f:
                    idx_to_class  = {v: k for k, v in json.load(f).items()}
                    disease_classes = [idx_to_class[i] for i in range(len(idx_to_class))]
            print(f"Disease model loaded ({len(disease_classes)} classes)")
        except Exception as e:
            print(f"Disease model load failed: {e}")
    else:
        print("Disease model not found. Run train_disease_model.py")
    print("All models loaded — Smart Crop Advisor v2 ready!")

class CropRequest(BaseModel):
    nitrogen:    float = Field(..., ge=0, le=200)
    phosphorus:  float = Field(..., ge=0, le=200)
    potassium:   float = Field(..., ge=0, le=200)
    temperature: float = Field(..., ge=0, le=50)
    humidity:    float = Field(..., ge=0, le=100)
    ph:          float = Field(..., ge=0, le=14)
    rainfall:    float = Field(..., ge=0, le=500)
    land_acres:  Optional[float] = Field(1.0)
    language:    Optional[str]   = Field("en")

class FertilizerRequest(BaseModel):
    temperature: float
    humidity:    float
    moisture:    float
    soil_type:   str
    crop_type:   str
    nitrogen:    float
    potassium:   float
    phosphorous: float
    language:    Optional[str] = Field("en")

class ChatRequest(BaseModel):
    message:  str
    history:  Optional[List[dict]] = Field(default_factory=list)
    language: Optional[str]        = Field("en")
    context:  Optional[dict]       = Field(default_factory=dict)
    api_key:  Optional[str]        = Field("")

class TranslateRequest(BaseModel):
    text:   str
    target: str

MARKET_PRICES = {
    "rice":        {"price_per_quintal": 2183,  "cultivation_cost_per_acre": 12000},
    "wheat":       {"price_per_quintal": 2275,  "cultivation_cost_per_acre": 10000},
    "maize":       {"price_per_quintal": 1962,  "cultivation_cost_per_acre": 8000},
    "chickpea":    {"price_per_quintal": 5440,  "cultivation_cost_per_acre": 10000},
    "kidneybeans": {"price_per_quintal": 6000,  "cultivation_cost_per_acre": 11000},
    "pigeonpeas":  {"price_per_quintal": 7000,  "cultivation_cost_per_acre": 9000},
    "mothbeans":   {"price_per_quintal": 5500,  "cultivation_cost_per_acre": 7000},
    "mungbean":    {"price_per_quintal": 7196,  "cultivation_cost_per_acre": 8500},
    "blackgram":   {"price_per_quintal": 6600,  "cultivation_cost_per_acre": 8000},
    "lentil":      {"price_per_quintal": 5500,  "cultivation_cost_per_acre": 9000},
    "pomegranate": {"price_per_quintal": 8000,  "cultivation_cost_per_acre": 25000},
    "banana":      {"price_per_quintal": 1500,  "cultivation_cost_per_acre": 30000},
    "mango":       {"price_per_quintal": 4000,  "cultivation_cost_per_acre": 20000},
    "grapes":      {"price_per_quintal": 6000,  "cultivation_cost_per_acre": 35000},
    "watermelon":  {"price_per_quintal": 1000,  "cultivation_cost_per_acre": 15000},
    "muskmelon":   {"price_per_quintal": 1200,  "cultivation_cost_per_acre": 14000},
    "apple":       {"price_per_quintal": 8000,  "cultivation_cost_per_acre": 40000},
    "orange":      {"price_per_quintal": 3500,  "cultivation_cost_per_acre": 18000},
    "papaya":      {"price_per_quintal": 1500,  "cultivation_cost_per_acre": 20000},
    "coconut":     {"price_per_quintal": 2500,  "cultivation_cost_per_acre": 22000},
    "cotton":      {"price_per_quintal": 6620,  "cultivation_cost_per_acre": 15000},
    "jute":        {"price_per_quintal": 5050,  "cultivation_cost_per_acre": 8000},
    "coffee":      {"price_per_quintal": 12000, "cultivation_cost_per_acre": 30000},
    "tomato":      {"price_per_quintal": 1500,  "cultivation_cost_per_acre": 18000},
    "onion":       {"price_per_quintal": 2000,  "cultivation_cost_per_acre": 12000},
    "potato":      {"price_per_quintal": 1200,  "cultivation_cost_per_acre": 10000},
    "soybean":     {"price_per_quintal": 3900,  "cultivation_cost_per_acre": 11000},
    "sugarcane":   {"price_per_quintal": 3150,  "cultivation_cost_per_acre": 20000},
    "groundnut":   {"price_per_quintal": 5500,  "cultivation_cost_per_acre": 14000},
}

YIELD_PER_ACRE = {
    "rice": 14,"wheat": 16,"maize": 15,"chickpea": 6,"kidneybeans": 5,"pigeonpeas": 6,
    "mothbeans": 4,"mungbean": 5,"blackgram": 5,"lentil": 5,"pomegranate": 60,"banana": 120,
    "mango": 40,"grapes": 80,"watermelon": 100,"muskmelon": 80,"apple": 50,"orange": 60,
    "papaya": 100,"coconut": 50,"cotton": 8,"jute": 15,"coffee": 6,"tomato": 80,"onion": 60,
    "potato": 70,"soybean": 10,"sugarcane": 300,"groundnut": 8,
}

DISEASE_TREATMENT = {
    "Pepper__bell___Bacterial_spot":               "Apply copper-based bactericide. Remove infected leaves. Avoid overhead irrigation.",
    "Pepper__bell___healthy":                      "Plant is healthy. Continue regular monitoring.",
    "Potato___Early_blight":                       "Apply mancozeb or chlorothalonil fungicide. Ensure proper crop rotation.",
    "Potato___Late_blight":                        "Apply metalaxyl-based fungicide immediately. Destroy infected plants.",
    "Potato___healthy":                            "Plant is healthy. Continue regular monitoring.",
    "Tomato__Target_Spot":                         "Apply chlorothalonil or azoxystrobin fungicide. Improve air circulation.",
    "Tomato__Tomato_mosaic_virus":                 "No chemical cure. Remove and destroy infected plants. Control aphids.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus":       "Control whitefly vectors. Use reflective mulches. Remove infected plants.",
    "Tomato_Bacterial_spot":                       "Apply copper hydroxide spray. Use disease-free seeds.",
    "Tomato_Early_blight":                         "Apply mancozeb fungicide. Remove lower infected leaves.",
    "Tomato_healthy":                              "Plant is healthy. Continue regular monitoring.",
    "Tomato_Late_blight":                          "Apply metalaxyl fungicide. Avoid wet foliage. Remove infected debris.",
    "Tomato_Leaf_Mold":                            "Improve ventilation. Apply copper fungicide or trifloxystrobin.",
    "Tomato_Septoria_leaf_spot":                   "Apply mancozeb or copper fungicide. Mulch soil. Avoid overhead watering.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Apply miticide/acaricide. Increase humidity. Introduce predatory mites.",
}

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "models_loaded": crop_model is not None}

@app.get("/api/languages")
async def languages():
    return {"languages": get_supported_languages()}

@app.post("/api/predict/crop")
async def predict_crop(req: CropRequest):
    if crop_model is None:
        raise HTTPException(status_code=503, detail="Crop model not loaded")
    features        = np.array([[req.nitrogen, req.phosphorus, req.potassium,
                                  req.temperature, req.humidity, req.ph, req.rainfall]])
    features_scaled = crop_scaler.transform(features)
    probs           = crop_model.predict_proba(features_scaled)[0]
    top3            = np.argsort(probs)[::-1][:3]
    recs = []
    for idx in top3:
        crop   = crop_le.classes_[idx]
        conf   = round(float(probs[idx]) * 100, 2)
        mkt    = MARKET_PRICES.get(crop, {})
        yld    = YIELD_PER_ACRE.get(crop, 10)
        price  = mkt.get("price_per_quintal", 3000)
        cost   = mkt.get("cultivation_cost_per_acre", 10000)
        rev    = yld * price * req.land_acres
        profit = rev - (cost * req.land_acres)
        recs.append({
            "crop": crop, "confidence_pct": conf,
            "market_price_per_quintal": price,
            "expected_yield_quintal_per_acre": yld,
            "cultivation_cost_per_acre": cost,
            "estimated_revenue_inr": round(rev),
            "estimated_profit_inr":  round(profit),
            "profit_margin_pct": round((profit / rev) * 100, 1) if rev > 0 else 0,
        })
    recs.sort(key=lambda x: x["estimated_profit_inr"], reverse=True)
    advice = (f"Based on your soil and weather conditions, {recs[0]['crop'].title()} "
              f"is the best crop with estimated profit of Rs.{recs[0]['estimated_profit_inr']:,}.")
    if req.language != "en":
        advice = translate_text(advice, req.language)
    return {"status": "success", "top_recommendations": recs, "land_acres": req.land_acres, "advice": advice}

@app.post("/api/predict/fertilizer")
async def predict_fertilizer(req: FertilizerRequest):
    if fert_model is None:
        raise HTTPException(status_code=503, detail="Fertilizer model not loaded")
    try:
        soil_enc = fert_le_soil.transform([req.soil_type])[0]
        crop_enc = fert_le_crop.transform([req.crop_type])[0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    features = np.array([[req.temperature, req.humidity, req.moisture,
                           soil_enc, crop_enc, req.nitrogen, req.potassium, req.phosphorous]])
    fs    = fert_scaler.transform(features)
    probs = fert_model.predict_proba(fs)[0]
    top3  = np.argsort(probs)[::-1][:3]
    recs  = [{"fertilizer": fert_le_fert.classes_[i],
               "confidence_pct": round(float(probs[i]) * 100, 2)} for i in top3]
    advice = f"For {req.crop_type} on {req.soil_type} soil, we recommend {recs[0]['fertilizer']} fertilizer."
    if req.language != "en":
        advice = translate_text(advice, req.language)
    return {"status": "success", "top_recommendations": recs,
            "valid_soil_types": list(fert_le_soil.classes_),
            "valid_crop_types": list(fert_le_crop.classes_), "advice": advice}

@app.post("/api/predict/disease")
async def predict_disease(file: UploadFile = File(...)):
    if disease_model is None:
        raise HTTPException(status_code=503, detail="Disease model not loaded. Run train_disease_model.py first.")
    contents = await file.read()
    h, w = disease_model.input_shape[1], disease_model.input_shape[2]
    img  = Image.open(io.BytesIO(contents)).convert("RGB").resize((w, h))
    arr  = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds   = disease_model.predict(arr)[0]
    top3    = np.argsort(preds)[::-1][:3]
    results = [{"disease": disease_classes[i],
                "confidence_pct": round(float(preds[i]) * 100, 2),
                "treatment": DISEASE_TREATMENT.get(disease_classes[i], "Consult local agricultural expert.")}
               for i in top3]
    return {"status": "success", "is_healthy": "healthy" in results[0]["disease"].lower(),
            "predictions": results, "image_name": file.filename}

@app.get("/api/market/price")
async def get_market_price(crop: str):
    crop = crop.lower()
    if crop not in MARKET_PRICES:
        raise HTTPException(status_code=404, detail=f"Crop '{crop}' not found.")
    d = MARKET_PRICES[crop]
    return {"crop": crop, "price_per_quintal_inr": d["price_per_quintal"],
            "cultivation_cost_per_acre_inr": d["cultivation_cost_per_acre"],
            "expected_yield_quintal_per_acre": YIELD_PER_ACRE.get(crop, 10), "source": "MSP 2024-25"}

@app.get("/api/market/forecast")
async def market_forecast(crop: str, language: str = "en"):
    result = get_forecast(crop)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    if language != "en":
        result["trend_advice"] = translate_text(result["trend_advice"], language)
    return result

@app.get("/api/market/forecast/crops")
async def forecast_crops():
    return {"crops": get_all_crops()}

@app.get("/api/weather")
async def get_weather(lat: float, lon: float, api_key: str = "demo"):
    if api_key in ("demo", "YOUR_KEY", ""):
        return {"location": "Bengaluru", "temperature": 27.0, "humidity": 68,
                "description": "partly cloudy", "rainfall_mm": 0, "wind_speed": 12,
                "note": "Add OpenWeather API key in Profile for real weather"}
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="Weather fetch failed.")
    d = r.json()
    return {"location": d.get("name", "Unknown"), "temperature": d["main"]["temp"],
            "humidity": d["main"]["humidity"], "description": d["weather"][0]["description"],
            "rainfall_mm": d.get("rain", {}).get("1h", 0),
            "wind_speed": d.get("wind", {}).get("speed", 0)}

@app.get("/api/weather/forecast")
async def weather_forecast_endpoint(lat: float, lon: float, api_key: str = "demo", language: str = "en"):
    if api_key in ("demo", "YOUR_KEY", ""):
        return {"days": [], "note": "Add OpenWeather API key in Profile for 7-day forecast"}
    try:
        result = await get_7day_forecast(lat, lon, api_key)
        if language != "en":
            for day in result["days"]:
                day["alerts"] = [translate_text(a, language) for a in day.get("alerts", [])]
        return result
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        response = await get_chat_response(
            message=req.message,
            history=req.history or [],
            language=req.language or "en",
            context=req.context or {},
            api_key=req.api_key or ""
        )
        return {"status": "success", "response": response, "language": req.language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/translate")
async def translate(req: TranslateRequest):
    translated = translate_text(req.text, req.target)
    return {"original": req.text, "translated": translated, "language": req.target}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)